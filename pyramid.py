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
the most text-relevant tokens, plus a random share of the rest.

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


def downsample_flat(X:torch.Tensor,shape:tuple,size:tuple,device,chunk:int=4096)->torch.Tensor:
    '''
    (N,C*H*W) bank points, possibly on the CPU in float16 -> (N,C*h*w) float32 on device
    '''
    C,H,W=shape
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
                     heatmap_lambda:float=0.0,keep:float=1.0,random_frac:float=0.0,generator:torch.Generator=None)->dict:
    '''
    every stride-1 p x p patch of images (K,C,H,W), each with an additive logit bias
        w * log_prior[parent] + heatmap_lambda * standardized heat score
    heat: (K,g,g) text heatmap on each image's CLIP token grid; a patch's score is the overlap-weighted mean
    heat of the tokens it covers (token_to_patches)
    keep < 1 keeps that fraction of the tokens, those with the highest heat over all K images, and then every
    patch that shares at least one pixel with a kept token, plus random_frac of the other patches (patches the
    text heatmap calls irrelevant still have to be available: every output region snaps to some patch, so a bank
    of only relevant patches tiles them everywhere)
    returns {patches (M,C*p*p), bias (M,) or None, y_sq (M,), n_total}
    '''
    K=images.shape[0]
    patches=F.unfold(images,p).transpose(1,2)
    L=patches.shape[1]
    patches=patches.reshape(K*L,-1)
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
        covered=token_to_patches(kept_tokens.reshape(heat.shape),size,p).flatten()>1e-6
        rest=(~covered).nonzero().squeeze(1)
        n_rand=int(round(random_frac*rest.numel()))
        perm=torch.randperm(rest.numel(),generator=generator,device=generator.device if generator is not None else "cpu")
        idx=torch.cat([covered.nonzero().squeeze(1),rest[perm[:n_rand].to(rest.device)]])
        patches=patches[idx]
        bias=None if bias is None else bias[idx]
    return {"patches":patches,"bias":bias,"y_sq":patches.pow(2).sum(-1),"n_total":K*L}


def patch_denoise(q:torch.Tensor,bank:dict,t:float,max_elements:int=2**27)->torch.Tensor:
    '''
    closed-form denoiser over the patch bank: softmax_j( -||q - t y_j||^2 / (2(1-t)^2) + bias_j ) weighted mean
    of the clean patches y_j (the clean estimate, not t*y), in query chunks of at most max_elements logits
    q: (Q,P) noisy patches -> (Q,P)
    '''
    Y,bias,y_sq=bank["patches"],bank["bias"],bank["y_sq"]
    chunk=max(1,max_elements//Y.shape[0])
    out=torch.empty_like(q)
    for i in range(0,q.shape[0],chunk):
        logits=(2*t*q[i:i+chunk]@Y.T-t*t*y_sq)/(2*(1-t)**2)
        if bias is not None:
            logits=logits+bias
        out[i:i+chunk]=torch.softmax(logits,dim=-1)@Y
    return out


def image_denoise(z:torch.Tensor,bank:dict,t:float,p:int,window:torch.Tensor,max_elements:int=2**27)->torch.Tensor:
    '''
    z: (B,C,H,W) -> clean estimate (B,C,H,W): denoise every stride-1 patch, then fold the estimates back as a
    window-weighted average over the patches covering each pixel
    '''
    B,C,H,W=z.shape
    q=F.unfold(z,p).transpose(1,2)
    L=q.shape[1]
    est=patch_denoise(q.reshape(B*L,-1),bank,t,max_elements).reshape(B,L,-1)*window
    num=F.fold(est.transpose(1,2),(H,W),p)
    den=F.fold(window[None,:,None].expand(1,-1,L),(H,W),p)
    return num/den


def blend(fine:torch.Tensor,coarser:torch.Tensor)->torch.Tensor:
    '''
    Laplacian two-scale blend: fine's detail band (fine - up(down(fine))) on top of up(coarser), where coarser is
    the finished level below
    '''
    size=fine.shape[-2:]
    return resize(coarser,size)+fine-resize(resize(fine,coarser.shape[-2:]),size)


@torch.no_grad()
def sample_level(n:int,bank:dict,shape:tuple,p:int,coarser:torch.Tensor,steps:int=10,rho:float=0.2,eta:float=0.0,
                 generator:torch.Generator=None,max_elements:int=2**27)->torch.Tensor:
    '''
    one patch level from pure noise, DDIM-style (eta=0 deterministic, eta=1 fresh noise every step), with the
    noise estimated before the blend and no blend on the last step (as in efficient-SID's uncond sampler)
    shape: (C,H,W) of this level, coarser: (n,C,h,w) finished level below -> (n,C,H,W)
    '''
    device=bank["patches"].device
    window=gaussian_window(shape[0],p,rho).to(device)
    z=torch.randn((n,*shape),generator=generator,device=device)
    ts=torch.linspace(0.0,1.0,steps+1).tolist()
    for t,t_next in zip(ts[:-1],ts[1:]):
        x_hat=image_denoise(z,bank,t,p,window,max_elements)
        if t_next>=1.0:
            z=x_hat
            break
        eps=(z-t*x_hat)/(1-t)
        x_hat=blend(x_hat,coarser)
        noise_scale=1-t_next
        if eta>0:
            fresh=torch.randn(z.shape,generator=generator,device=device)
            z=t_next*x_hat+math.sqrt(1-eta**2)*noise_scale*eps+eta*noise_scale*fresh
        else:
            z=t_next*x_hat+noise_scale*eps
    return z


def build_pyramid_banks(images:torch.Tensor,coarse_size:int,p:int,log_prior:torch.Tensor=None,w:float=1.0,
                        heat:torch.Tensor=None,heatmap_lambda:float=0.0,keep:float=1.0,random_frac:float=0.0,
                        generator:torch.Generator=None)->list:
    '''
    one patch bank per patch level (every level but the coarsest), finest first
    images: (K,C,H,W) the images supplying patches, heat: (K,g,g) on their CLIP token grids or None; the same
    tokens are kept at every level, each mapping to the patches that overlap it there
    '''
    sizes=pyramid_sizes(images.shape[-2],images.shape[-1],coarse_size)
    banks=[]
    for size in sizes[:-1]:
        assert min(size)>=p,f"patch_size {p} is larger than the {size[0]}x{size[1]} level"
        banks.append(build_patch_bank(resize(images,size),p,log_prior,w,heat,heatmap_lambda,keep,random_frac,generator))
    return banks


@torch.no_grad()
def sample_pyramid(n:int,coarse_X:torch.Tensor,shape:tuple,banks:list,coarse_size:int,p:int,log_prior:torch.Tensor=None,
                   w:float=1.0,sigma:float=0.0,M:int=1,cfdm_steps:int=50,start_t:float=0.0,steps:int=10,rho:float=0.2,
                   eta:float=0.0,generator:torch.Generator=None,max_elements:int=2**27,return_levels:bool=False):
    '''
    coarse_X: (N,C*hc*wc) bank points at the coarsest level (downsample_flat), with log_prior (N,) for the CFDM
    there; sigma is the absolute CFDM smoothing; shape: (C,H,W) at full resolution; banks: build_pyramid_banks
    returns (n,C*H*W), and with return_levels also every finished level, coarsest first
    '''
    C,H,W=shape
    sizes=pyramid_sizes(H,W,coarse_size)
    assert len(banks)==len(sizes)-1,f"{len(banks)} banks for {len(sizes)} levels"
    x=sample(n,coarse_X,log_prior,w=w,sigma=sigma,M=M,steps=cfdm_steps,start_t=start_t,generator=generator)
    x=x.reshape(n,C,*sizes[-1])
    levels=[x]
    for size,bank in zip(reversed(sizes[:-1]),reversed(banks)):
        x=sample_level(n,bank,(C,*size),p,x,steps,rho,eta,generator,max_elements)
        levels.append(x)
    if return_levels:
        return x.flatten(1),levels
    return x.flatten(1)
