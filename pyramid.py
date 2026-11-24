'''
coarse-to-fine closed-form sampling: the smoothed CFDM (cfdm.py, Scarvelis et al.) lays out a new image at the
coarsest pyramid level, where averaging a few training images is still plausible, and every finer level adds
real detail with the closed-form patch denoiser of Qiu et al. ("Efficient and training-free single-image
diffusion models", the efficient-SID-code submodule), run over the stride-1 patches of the prompt's retrieved
images instead of a single image. The bank's CLIP token embeddings (clip_tokens.py) tie the two together: every
CLIP token covers a g x g cell of the image, and maps at each level to the denoiser patches that overlap it.

Each patch level is a full trajectory from noise. After every denoiser call the clean estimate keeps only its
own detail band and takes its low frequencies from the finished coarser level (their Laplacian two-scale
blend), so the patches can sharpen the CFDM layout but never move it. Each candidate patch can carry a bias:
w * log p(parent image | prompt), plus heatmap_lambda * its standardized CLIP text-heatmap score (the
overlap-weighted mean of the heat of the tokens it covers), and the bank can be cut to the patches that overlap
the most text-relevant tokens (by any pixel, or by at least a set fraction of the patch), plus a random share of
the rest.

Every output is an exact convex combination of training data, which is what makes it attributable: the coarse
CFDM sample is sum_i w_i x_i (cfdm.sample), and the finest level's last step is a window-weighted fold of
softmax averages of training patches, so each output pixel is exactly a weighted sum of training pixels: of the
pixel at the same offset inside every source patch of every query patch covering it. return_attribution gives,
at the finest level, each pixel's exact share from each patch image, its top source pixels (image, y, x, weight;
pixel_sources), and the CFDM weights behind the layout.

Same time convention as cfdm.py: z_t = t x + (1-t) eps, t=0 is noise and t=1 is clean.
'''
import math
import torch
import torch.nn.functional as F
from cfdm import sample


def resize(x:torch.Tensor,size:tuple)->torch.Tensor:
    # (B,C,H,W) -> (B,C,*size); antialiased when shrinking
    return F.interpolate(x,size=tuple(size),mode="bicubic",antialias=True,align_corners=False)


def pyramid_sizes(H:int,W:int,coarse_size:int)->list:
    '''
    (H,W) of every level, finest first: halving each side while the short side stays >= coarse_size
    '''
    assert min(H,W)>=coarse_size,f"coarse_size {coarse_size} is larger than the {H}x{W} image"
    sizes=[(H,W)]
    while min(sizes[-1])//2>=coarse_size:
        h,w=sizes[-1]
        sizes.append((math.ceil(h/2),math.ceil(w/2)))
    return sizes


def downsample_flat(X:torch.Tensor,shape:tuple,size:tuple,device,chunk:int=0)->torch.Tensor:
    '''
    (N,C*H*W) bank points, possibly on the CPU in float16 -> (N,C*h*w) float32 on device; chunk: rows moved to the
    device at a time (0: about 512MB of them)
    '''
    C,H,W=shape
    chunk=chunk or max(1,2**27//(C*H*W))
    return torch.cat([resize(X[i:i+chunk].to(device).float().reshape(-1,C,H,W),size).flatten(1)
                      for i in range(0,X.shape[0],chunk)])


def gaussian_window(C:int,p:int,rho:float)->torch.Tensor:
    '''
    separable Gaussian over the patch's [-1,1] coordinates, in F.unfold's (C,p,p) order: (C*p*p,); small rho
    folds overlapping patches back mostly by their centers
    '''
    g=torch.exp(-0.5*(torch.linspace(-1,1,p)/rho)**2)
    return torch.outer(g,g).repeat(C,1,1).flatten()


def token_overlap(n:int,g:int,p:int)->torch.Tensor:
    '''
    (n-p+1,g): along one side of an n-pixel level whose image is split into g CLIP token rows (or columns), the
    number of pixels each stride-1 p-pixel patch window shares with each token row. Patch (y,x) and token (i,j)
    share O_y[y,i]*O_x[x,j] pixels
    '''
    start=torch.arange(n-p+1,dtype=torch.float64)[:,None]
    edges=torch.arange(g+1,dtype=torch.float64)*n/g
    return (torch.minimum(start+p,edges[1:])-torch.maximum(start,edges[:-1])).clamp_min(0).float()


def token_to_patches(token:torch.Tensor,size:tuple,p:int)->torch.Tensor:
    '''
    token: (K,g,g) values on the CLIP token grid -> (K,L): for every stride-1 p x p patch of a size=(H,W) level
    (F.unfold's position order), the sum of the tokens it overlaps, each weighted by the shared pixel count / p^2;
    a heatmap becomes each patch's mean heat, a 0/1 token mask its covered fraction
    '''
    g=token.shape[-1]
    O_y=token_overlap(size[0],g,p).to(token)
    O_x=token_overlap(size[1],g,p).to(token)
    return torch.einsum("yi,kij,xj->kyx",O_y,token,O_x).flatten(1)/(p*p)


def build_patch_bank(images:torch.Tensor,p:int,log_prior:torch.Tensor=None,w:float=1.0,heat:torch.Tensor=None,
                     heatmap_lambda:float=0.0,keep:float=1.0,random_frac:float=0.0,generator:torch.Generator=None,
                     min_overlap:float=0.0)->dict:
    '''
    every stride-1 p x p patch of images (K,C,H,W), each with an additive logit bias
        w * log_prior[parent] + heatmap_lambda * standardized heat score
    heat: (K,g,g) text heatmap on each image's CLIP token grid; a patch's score is the overlap-weighted mean
    heat of the tokens it covers (token_to_patches)
    keep < 1 keeps that fraction of the tokens, those with the highest heat over all K images, and then every
    patch with more than min_overlap of its pixels inside kept tokens (0: a single pixel is enough; 0.5: at least
    half the patch), plus random_frac of the other patches (patches the
    text heatmap calls irrelevant still have to be available: every output region snaps to some patch, so a bank
    of only relevant patches tiles them everywhere)
    returns {patches (M,C*p*p), bias (M,) or None, y_sq (M,), parent (M,) the image each patch comes from,
    pos (M,) its top-left corner as y*(W-p+1)+x, n_images K, n_total}
    '''
    K=images.shape[0]
    patches=F.unfold(images,p).transpose(1,2)
    L=patches.shape[1]
    patches=patches.reshape(K*L,-1)
    parent=torch.arange(K,device=patches.device).repeat_interleave(L)
    pos=torch.arange(L,device=patches.device).repeat(K)
    bias=None
    if log_prior is not None and w!=0:
        bias=w*log_prior.to(patches).repeat_interleave(L)
    size=images.shape[-2:]
    if heat is not None and heatmap_lambda!=0:
        score=token_to_patches(heat.to(patches),size,p).flatten()
        score=(score-score.mean())/score.std().clamp_min(1e-8)
        bias=heatmap_lambda*score if bias is None else bias+heatmap_lambda*score
    if keep<1:
        assert heat is not None,"heatmap filtering needs heat"
        flat=heat.to(patches).flatten()
        kept_tokens=torch.zeros_like(flat)
        kept_tokens[torch.topk(flat,max(1,math.ceil(keep*flat.numel()))).indices]=1
        # the fraction of each patch's pixels inside kept tokens
        covered_frac=token_to_patches(kept_tokens.reshape(heat.shape),size,p).flatten()
        # capped at the best-covered patch: at coarse levels, where patches span several tokens, no patch may
        # reach min_overlap, and an empty level can't denoise anything
        covered=covered_frac>1e-6 if min_overlap<=0 else covered_frac>=min(min_overlap,covered_frac.max().item())-1e-6
        rest=(~covered).nonzero().squeeze(1)
        n_rand=int(round(random_frac*rest.numel()))
        perm=torch.randperm(rest.numel(),generator=generator,device=generator.device if generator is not None else "cpu")
        idx=torch.cat([covered.nonzero().squeeze(1),rest[perm[:n_rand].to(rest.device)]])
        patches=patches[idx]
        parent=parent[idx]
        pos=pos[idx]
        bias=None if bias is None else bias[idx]
    return {"patches":patches,"bias":bias,"y_sq":patches.pow(2).sum(-1),"parent":parent,"pos":pos,"n_images":K,"n_total":K*L}


def patch_denoise(q:torch.Tensor,bank:dict,t:float,max_elements:int=2**27,return_attribution:bool=False,
                  topk:int=8):
    '''
    closed-form denoiser over the patch bank: softmax_j( -||q - t y_j||^2 / (2(1-t)^2) + bias_j ) weighted mean
    of the clean patches y_j (the clean estimate, not t*y), in query chunks of at most max_elements logits
    q: (Q,P) noisy patches -> (Q,P); return_attribution also returns {mass (Q,K): the softmax mass on each patch
    image, top_weight/top_index (Q,topk): the largest softmax weights and their bank patches}
    '''
    Y,bias,y_sq=bank["patches"],bank["bias"],bank["y_sq"]
    chunk=max(1,max_elements//Y.shape[0])
    out=torch.empty_like(q)
    if return_attribution:
        topk=min(topk,Y.shape[0])
        attr={"mass":torch.zeros(q.shape[0],bank["n_images"],device=q.device,dtype=q.dtype),
              "top_weight":torch.empty(q.shape[0],topk,device=q.device,dtype=q.dtype),
              "top_index":torch.empty(q.shape[0],topk,device=q.device,dtype=torch.long)}
    for i in range(0,q.shape[0],chunk):
        logits=(2*t*q[i:i+chunk]@Y.T-t*t*y_sq)/(2*(1-t)**2)
        if bias is not None:
            logits=logits+bias
        weights=torch.softmax(logits,dim=-1)
        out[i:i+chunk]=weights@Y
        if return_attribution:
            attr["mass"][i:i+chunk].index_add_(1,bank["parent"],weights)
            top=torch.topk(weights,topk,dim=-1)
            attr["top_weight"][i:i+chunk]=top.values
            attr["top_index"][i:i+chunk]=top.indices
    return (out,attr) if return_attribution else out


def pixel_sources(top_weight:torch.Tensor,top_index:torch.Tensor,bank:dict,size:tuple,p:int,spatial:torch.Tensor,
                  n_sources:int=4)->dict:
    '''
    one image's top source pixels. Output pixel (y,x) is offset (dy,dx) of the query patch at (y-dy,x-dx), whose
    estimate weights bank patch j by alpha_j, and that patch's pixel at the same offset is (pos_j + (dy,dx)) of its
    parent image; so the pixel's weight on that source pixel is window(dy,dx) * alpha_j / sum of the windows over
    the patches covering (y,x), summed over every way of reaching it (a copied region reaches one source pixel
    from all p^2 patches). Exact up to each query patch keeping only its top bank patches
    top_weight/top_index: (L,r) from patch_denoise, L the (H-p+1)*(W-p+1) query patches
    returns {image, y, x: (H,W,n_sources) long source pixels (image -1: no source), weight (H,W,n_sources)}
    '''
    H,W=size
    Hl,Wl=H-p+1,W-p+1
    r=top_weight.shape[-1]
    grid=lambda v:v.reshape(Hl,Wl,r).permute(2,0,1)
    weight=grid(top_weight)
    image=grid(bank["parent"][top_index])
    pos=bank["pos"][top_index]
    sy,sx=grid(pos//Wl),grid(pos%Wl)
    weights,keys=[],[]
    for dy in range(p):
        for dx in range(p):
            # shift the query-patch grid onto the pixels at offset (dy,dx) of each patch
            pad=(dx,W-Wl-dx,dy,H-Hl-dy)
            weights.append(F.pad(weight*spatial[dy*p+dx],pad))
            keys.append(F.pad(image*(H*W)+(sy+dy)*W+(sx+dx),pad,value=-1))
    den=sum(F.pad(torch.full((Hl,Wl),float(spatial[dy*p+dx]),device=weight.device),(dx,W-Wl-dx,dy,H-Hl-dy))
            for dy in range(p) for dx in range(p))
    weights=(torch.cat(weights)/den).reshape(-1,H*W).T
    keys=torch.cat(keys).reshape(-1,H*W).T
    # merge the candidates that reach the same source pixel: sort each pixel's keys and sum each run
    keys,order=keys.sort(dim=1)
    weights=weights.gather(1,order)
    new=torch.ones_like(keys,dtype=torch.bool)
    new[:,1:]=keys[:,1:]!=keys[:,:-1]
    run=new.cumsum(1)-1
    totals=torch.zeros_like(weights).scatter_add_(1,run,weights)
    run_keys=torch.full_like(keys,-1).scatter_(1,run,keys)
    top=torch.topk(totals,min(n_sources,totals.shape[1]),dim=1)
    key=run_keys.gather(1,top.indices)
    valid=(key>=0)&(top.values>0)
    key=key.clamp_min(0)
    out={"image":torch.where(valid,key//(H*W),-1),"y":(key%(H*W))//W,"x":key%W,"weight":top.values*valid}
    return {k:v.reshape(H,W,-1) for k,v in out.items()}


def image_denoise(z:torch.Tensor,bank:dict,t:float,p:int,window:torch.Tensor,max_elements:int=2**27,
                  return_attribution:bool=False,topk:int=8,n_sources:int=4):
    '''
    z: (B,C,H,W) -> clean estimate (B,C,H,W): denoise every stride-1 patch, then fold the estimates back as a
    window-weighted average over the patches covering each pixel
    return_attribution also returns {maps (B,K,H,W): each pixel's exact share from each patch image, the same
    window-weighted fold of the patches' softmax mass per image (sums to 1 over K at every pixel); sources: each
    pixel's top n_sources source pixels (pixel_sources), stacked over B}
    '''
    B,C,H,W=z.shape
    q=F.unfold(z,p).transpose(1,2)
    L=q.shape[1]
    est=patch_denoise(q.reshape(B*L,-1),bank,t,max_elements,return_attribution,topk)
    if return_attribution:
        est,attr=est
        mass=attr["mass"]
    est=est.reshape(B,L,-1)*window
    num=F.fold(est.transpose(1,2),(H,W),p)
    den=F.fold(window[None,:,None].expand(1,-1,L),(H,W),p)
    if not return_attribution:
        return num/den
    # the window is the same p x p one on every channel
    spatial=window[:p*p]
    K=mass.shape[-1]
    mass=mass.reshape(B,L,K)
    # one sample at a time: the (K*p*p, L) fold input is large at 256px
    maps=torch.stack([F.fold((mass[b].T[:,None,:]*spatial[None,:,None]).reshape(1,K*p*p,L),(H,W),p)[0]
                      for b in range(B)])/den[:,:1]
    top_weight,top_index=attr["top_weight"].reshape(B,L,-1),attr["top_index"].reshape(B,L,-1)
    sources=[pixel_sources(top_weight[b],top_index[b],bank,(H,W),p,spatial,n_sources) for b in range(B)]
    sources={k:torch.stack([s[k] for s in sources]) for k in sources[0]}
    return num/den,{"maps":maps,"sources":sources}


def blend(fine:torch.Tensor,coarser:torch.Tensor)->torch.Tensor:
    '''
    Laplacian two-scale blend: fine's detail band (fine - up(down(fine))) on top of up(coarser), where coarser is
    the finished level below
    '''
    size=fine.shape[-2:]
    return resize(coarser,size)+fine-resize(resize(fine,coarser.shape[-2:]),size)


@torch.no_grad()
def sample_level(n:int,bank:dict,shape:tuple,p:int,coarser:torch.Tensor,steps:int=10,rho:float=0.2,eta:float=0.0,
                 generator:torch.Generator=None,max_elements:int=2**27,return_attribution:bool=False,
                 topk:int=8,n_sources:int=4):
    '''
    one patch level from pure noise, DDIM-style (eta=0 deterministic, eta=1 fresh noise every step), with the
    noise estimated before the blend and no blend on the last step (as in efficient-SID's uncond sampler)
    shape: (C,H,W) of this level, coarser: (n,C,h,w) finished level below -> (n,C,H,W)
    the unblended last step makes the output exactly the patch fold, so return_attribution also returns its
    attribution (image_denoise)
    '''
    device=bank["patches"].device
    window=gaussian_window(shape[0],p,rho).to(device)
    z=torch.randn((n,*shape),generator=generator,device=device)
    ts=torch.linspace(0.0,1.0,steps+1).tolist()
    maps=None
    for t,t_next in zip(ts[:-1],ts[1:]):
        if t_next>=1.0:
            z=image_denoise(z,bank,t,p,window,max_elements,return_attribution,topk,n_sources)
            if return_attribution:
                z,maps=z
            break
        x_hat=image_denoise(z,bank,t,p,window,max_elements)
        eps=(z-t*x_hat)/(1-t)
        x_hat=blend(x_hat,coarser)
        noise_scale=1-t_next
        if eta>0:
            fresh=torch.randn(z.shape,generator=generator,device=device)
            z=t_next*x_hat+math.sqrt(1-eta**2)*noise_scale*eps+eta*noise_scale*fresh
        else:
            z=t_next*x_hat+noise_scale*eps
    return (z,maps) if return_attribution else z


def build_pyramid_banks(images:torch.Tensor,coarse_size:int,p:int,log_prior:torch.Tensor=None,w:float=1.0,
                        heat:torch.Tensor=None,heatmap_lambda:float=0.0,keep:float=1.0,random_frac:float=0.0,
                        generator:torch.Generator=None,min_overlap:float=0.0)->list:
    '''
    one patch bank per patch level (every level but the coarsest), finest first
    images: (K,C,H,W) the images supplying patches, heat: (K,g,g) on their CLIP token grids or None; the same
    tokens are kept at every level, each mapping to the patches that overlap it there
    '''
    sizes=pyramid_sizes(images.shape[-2],images.shape[-1],coarse_size)
    banks=[]
    for size in sizes[:-1]:
        assert min(size)>=p,f"patch_size {p} is larger than the {size[0]}x{size[1]} level"
        banks.append(build_patch_bank(resize(images,size),p,log_prior,w,heat,heatmap_lambda,keep,random_frac,generator,min_overlap))
    return banks


@torch.no_grad()
def sample_pyramid(n:int,coarse_X:torch.Tensor,shape:tuple,banks:list,coarse_size:int,p:int,log_prior:torch.Tensor=None,
                   w:float=1.0,sigma:float=0.0,M:int=1,cfdm_steps:int=50,start_t:float=0.0,steps:int=10,rho:float=0.2,
                   eta:float=0.0,generator:torch.Generator=None,max_elements:int=2**27,return_levels:bool=False,
                   return_attribution:bool=False,topk:int=8,n_sources:int=4):
    '''
    coarse_X: (N,C*hc*wc) bank points at the coarsest level (downsample_flat), with log_prior (N,) for the CFDM
    there; sigma is the absolute CFDM smoothing; shape: (C,H,W) at full resolution; banks: build_pyramid_banks
    returns (n,C*H*W), with return_levels also every finished level, coarsest first, and with return_attribution
    also {"layout": (n,N) the CFDM's convex weights over coarse_X, "content": the finest level's image_denoise
    attribution (per-pixel shares of the K patch images and top source pixels), or None without patch levels}
    '''
    C,H,W=shape
    sizes=pyramid_sizes(H,W,coarse_size)
    assert len(banks)==len(sizes)-1,f"{len(banks)} banks for {len(sizes)} levels"
    x=sample(n,coarse_X,log_prior,w=w,sigma=sigma,M=M,steps=cfdm_steps,start_t=start_t,generator=generator,
             return_weights=return_attribution)
    attribution={"layout":None,"content":None}
    if return_attribution:
        x,attribution["layout"]=x
    x=x.reshape(n,C,*sizes[-1])
    levels=[x]
    for j,(size,bank) in enumerate(zip(reversed(sizes[:-1]),reversed(banks))):
        finest=j==len(banks)-1
        x=sample_level(n,bank,(C,*size),p,x,steps,rho,eta,generator,max_elements,return_attribution and finest,
                       topk,n_sources)
        if return_attribution and finest:
            x,attribution["content"]=x
        levels.append(x)
    out=(x.flatten(1),)+((levels,) if return_levels else ())+((attribution,) if return_attribution else ())
    return out if len(out)>1 else out[0]
