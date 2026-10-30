'''
Text-conditioned smoothed closed-form diffusion (Scarvelis et al. 2023, arXiv:2310.12395).

Unconditional model: rho_t(z) = 1/N sum_i N(z; t x_i, (1-t)^2 I).
Conditional model:   rho_t(z|c) = sum_i p(x_i|c) N(z; t x_i, (1-t)^2 I),
with p(x_i|c) = softmax_i(logit_scale * cos(c, tau_i)) from CLIP, tau_i the CLIP image
embedding of x_i. Since p(x_i|c) does not depend on z, the exact conditional score is

    grad log rho_t(z|c) = (k_t(z,c) - z) / (1-t)^2
    k_t(z,c) = sum_i softmax_i( -||z - t x_i||^2 / (2(1-t)^2) + w * log p(x_i|c) ) t x_i

w=0 is unconditional, w=1 is the exact conditional model, w>1 sharpens like CFG.
'''
import torch


def text_log_prior(text_embeds:torch.Tensor,image_embeds:torch.Tensor,logit_scale:float)->torch.Tensor:
    '''
    text_embeds: (P,E) or (E,), image_embeds: (N,E), both L2-normalized
    returns log p(x_i|c) of shape (P,N) or (N,)
    equivalent to -logit_scale/2 * ||c - tau_i||^2 up to a constant
    '''
    logits=logit_scale*text_embeds@image_embeds.T
    return torch.log_softmax(logits,dim=-1)


def top_k_subset(X:torch.Tensor,log_prior:torch.Tensor,k:int):
    '''
    keep the k training points with the highest text prior (log_prior: (N,))
    the prior is constant over the trajectory so this only needs to be done once per prompt
    returns X[idx], renormalized log_prior[idx], idx
    '''
    if k<=0 or k>=X.shape[0]:
        return X,log_prior,torch.arange(X.shape[0],device=X.device)
    idx=torch.topk(log_prior,k).indices
    return X[idx],torch.log_softmax(log_prior[idx],dim=-1),idx


def weight_logits(z:torch.Tensor,X:torch.Tensor,t:float,log_prior:torch.Tensor=None,w:float=1.0,x_sq:torch.Tensor=None)->torch.Tensor:
    '''
    z: (B,D), X: (N,D), log_prior: (N,) or (B,N)
    -||z - t x_i||^2/(2(1-t)^2) + w log p(x_i|c), dropping ||z||^2 since it is constant over i
    '''
    if x_sq is None:
        x_sq=X.pow(2).sum(-1)
    logits=(2*t*z@X.T-t*t*x_sq)/(2*(1-t)**2)
    if log_prior is not None and w!=0:
        logits=logits+w*log_prior
    return logits


def k_t(z:torch.Tensor,X:torch.Tensor,t:float,log_prior:torch.Tensor=None,w:float=1.0,x_sq:torch.Tensor=None)->torch.Tensor:
    '''
    softmax-weighted average of the rescaled training points t*x_i (Eq. 2, plus the text prior)
    '''
    weights=torch.softmax(weight_logits(z,X,t,log_prior,w,x_sq),dim=-1)
    return t*weights@X


def smoothed_k_t(z:torch.Tensor,X:torch.Tensor,t:float,log_prior:torch.Tensor=None,w:float=1.0,
                 sigma:float=0.0,M:int=1,x_sq:torch.Tensor=None,generator:torch.Generator=None)->torch.Tensor:
    '''
    1/M sum_m k_t(z + sigma eps_m) (Eq. 3); the text prior is not perturbed
    '''
    if sigma==0 or M<1:
        return k_t(z,X,t,log_prior,w,x_sq)
    total=torch.zeros_like(z)
    for _ in range(M):
        eps=torch.randn(z.shape,generator=generator,device=z.device,dtype=z.dtype)
        total+=k_t(z+sigma*eps,X,t,log_prior,w,x_sq)
    return total/M


def score(z:torch.Tensor,X:torch.Tensor,t:float,log_prior:torch.Tensor=None,w:float=1.0,
          sigma:float=0.0,M:int=1,x_sq:torch.Tensor=None,generator:torch.Generator=None)->torch.Tensor:
    return (smoothed_k_t(z,X,t,log_prior,w,sigma,M,x_sq,generator)-z)/(1-t)**2


def velocity(z:torch.Tensor,X:torch.Tensor,t:float,log_prior:torch.Tensor=None,w:float=1.0,
             sigma:float=0.0,M:int=1,x_sq:torch.Tensor=None,generator:torch.Generator=None)->torch.Tensor:
    '''
    v_t(z) = (z + (1-t) s_t(z)) / t (Eq. 6), written as (k - t z) / (t (1-t)) to avoid the (1-t)^2 blowup
    '''
    k=smoothed_k_t(z,X,t,log_prior,w,sigma,M,x_sq,generator)
    return (k-t*z)/(t*(1-t))


def initial_sample(n:int,X:torch.Tensor,start_t:float,log_prior:torch.Tensor=None,w:float=1.0,
                   generator:torch.Generator=None)->torch.Tensor:
    '''
    exact sample from rho_T(z|c): pick i with prob softmax(w log p(x_i|c)), then z = T x_i + (1-T) eps
    (Sec. 5, "fewer steps"); start_t=0 is plain N(0,I)
    '''
    noise=torch.randn((n,X.shape[1]),generator=generator,device=X.device,dtype=X.dtype)
    if start_t<=0:
        return noise
    if log_prior is None or w==0:
        probs=torch.full((X.shape[0],),1.0/X.shape[0],device=X.device)
    else:
        probs=torch.softmax(w*log_prior.float(),dim=-1)
    idx=torch.multinomial(probs,n,replacement=True,generator=generator)
    return start_t*X[idx]+(1-start_t)*noise


@torch.no_grad()
def sample(n:int,X:torch.Tensor,log_prior:torch.Tensor=None,w:float=1.0,sigma:float=0.0,M:int=1,
           steps:int=50,start_t:float=0.0,generator:torch.Generator=None,return_trajectory:bool=False,z0:torch.Tensor=None):
    '''
    forward Euler from start_t to t=1 (Algorithm 1). The last step from t to 1 lands exactly
    on smoothed_k_t(z)/t, so nothing blows up at t=1.
    X: (N,D) training points (flattened pixels or latents), log_prior: (N,) from text_log_prior
    z0: optional (n,D) starting point at start_t instead of initial_sample (e.g. shared noise with start_t ~ 0)
    '''
    assert 0<=start_t<1,"start_t must be in [0,1)"
    x_sq=X.pow(2).sum(-1)
    z=initial_sample(n,X,start_t,log_prior,w,generator) if z0 is None else z0
    ts=torch.linspace(start_t,1.0,steps+1).tolist()
    if ts[0]==0:
        # v_0 is undefined (1/t); k_0 = 0 so the first Euler step only rescales noise
        ts[0]=1e-4
    trajectory=[z]
    for t,t_next in zip(ts[:-1],ts[1:]):
        if t_next>=1.0:
            z=smoothed_k_t(z,X,t,log_prior,w,sigma,M,x_sq,generator)/t
        else:
            z=z+(t_next-t)*velocity(z,X,t,log_prior,w,sigma,M,x_sq,generator)
        if return_trajectory:
            trajectory.append(z)
    if return_trajectory:
        return z,trajectory
    return z


'''
multiscale patch CFDM: at scale s the (C,H,W) point is cut into an s x s grid of patches, and each
patch region gets its own closed-form k_t over candidate patches from the bank images, taken either
from every grid position (positions="any") or only from the same position (positions="same");
each candidate keeps its parent image's text prior (or, with patch_prior="none", candidates are weighted by
visual similarity alone: the k_t distance between the noisy patch and each candidate). The per-scale k_t's are
blended with weights alpha_s(t), so different regions can pull toward patches of different images.
Only scale 1 is the exact score of a distribution; the patch scales are a local-score heuristic.
'''


def patchify(x:torch.Tensor,shape:tuple,s:int)->torch.Tensor:
    '''
    x: (B,D) flattened (C,H,W) points -> (B,s*s,C*(H/s)*(W/s)), patches in row-major grid order
    '''
    C,H,W=shape
    assert H%s==0 and W%s==0,f"scale {s} does not divide {H}x{W}"
    h,w=H//s,W//s
    x=x.reshape(-1,C,s,h,s,w).permute(0,2,4,1,3,5)
    return x.reshape(x.shape[0],s*s,C*h*w)


def unpatchify(p:torch.Tensor,shape:tuple,s:int)->torch.Tensor:
    '''
    inverse of patchify: (B,s*s,P) -> (B,D)
    '''
    C,H,W=shape
    h,w=H//s,W//s
    p=p.reshape(-1,s,s,C,h,w).permute(0,3,1,4,2,5)
    return p.reshape(p.shape[0],C*H*W)


def patch_bank(X:torch.Tensor,shape:tuple,s:int,log_prior:torch.Tensor=None):
    '''
    candidate patches at scale s from every grid position of every image: (N*s*s,P),
    each with its parent image's log prior (renormalized, so every image keeps the same total mass)
    '''
    patches=patchify(X,shape,s).reshape(-1,X.shape[1]//(s*s))
    if log_prior is None:
        return patches,None
    return patches,torch.log_softmax(log_prior.repeat_interleave(s*s),dim=-1)


def scale_weights(t:float,n_scales:int,schedule:str="coarse_to_fine",start_t:float=0.0,constant:list=None,
                  switch_frac:float=0.5)->list:
    '''
    alpha_s(t), summing to 1. coarse_to_fine moves linearly (hat functions) from the first scale at
    start_t to the last scale at t=1; fine_to_coarse is the reverse; constant uses the given weights;
    switch uses only the first scale for the first switch_frac of [start_t,1], then only the last scale
    '''
    if n_scales==1:
        return [1.0]
    if schedule=="constant":
        total=sum(constant)
        return [c/total for c in constant]
    u=min(max((t-start_t)/(1-start_t),0.0),1.0)
    if schedule=="switch":
        return [1.0]+[0.0]*(n_scales-1) if u<switch_frac else [0.0]*(n_scales-1)+[1.0]
    if schedule=="fine_to_coarse":
        u=1-u
    pos=u*(n_scales-1)
    return [max(0.0,1-abs(pos-j)) for j in range(n_scales)]


def same_position_k_t(z_p:torch.Tensor,patches:torch.Tensor,t:float,log_prior:torch.Tensor=None,w:float=1.0,
                      sigma:float=0.0,M:int=1,x_sq:torch.Tensor=None,generator:torch.Generator=None)->torch.Tensor:
    '''
    smoothed k_t where region j only sees patch j of each bank image
    z_p: (B,R,P) query patches, patches: (N,R,P), log_prior: (N,) parent prior, x_sq: (N,R)
    '''
    n_draws=M if sigma>0 and M>=1 else 1
    total=torch.zeros_like(z_p)
    for _ in range(n_draws):
        q=z_p
        if sigma>0:
            q=z_p+sigma*torch.randn(z_p.shape,generator=generator,device=z_p.device,dtype=z_p.dtype)
        logits=(2*t*torch.einsum("brp,nrp->brn",q,patches)-t*t*x_sq.T[None])/(2*(1-t)**2)
        if log_prior is not None and w!=0:
            logits=logits+w*log_prior
        total+=t*torch.einsum("brn,nrp->brp",torch.softmax(logits,dim=-1),patches)
    return total/n_draws


def multiscale_k_t(z:torch.Tensor,banks:list,t:float,alphas:list,w:float=1.0,sigmas:list=None,M:int=1,
                   generator:torch.Generator=None)->torch.Tensor:
    '''
    sum_s alpha_s * k_t^(s)(z), where k_t^(s) runs a smoothed k_t independently on each patch region
    banks: list of dicts {s, shape, positions, patches, log_prior, x_sq} built in sample_multiscale
    '''
    out=torch.zeros_like(z)
    for j,(bank,alpha) in enumerate(zip(banks,alphas)):
        if alpha==0:
            continue
        s=bank["s"]
        sigma=0.0 if sigmas is None else sigmas[j]
        if bank.get("positions","any")=="same":
            k=same_position_k_t(patchify(z,bank["shape"],s),bank["patches"],t,bank["log_prior"],w,sigma,M,bank["x_sq"],generator)
        else:
            # every region of every sample is an independent query against all candidate patches
            z_p=patchify(z,bank["shape"],s).reshape(z.shape[0]*s*s,-1)
            k=smoothed_k_t(z_p,bank["patches"],t,bank["log_prior"],w,sigma,M,bank["x_sq"],generator)
        out+=alpha*unpatchify(k.reshape(z.shape[0],s*s,-1),bank["shape"],s)
    return out


def median_patch_distance(X:torch.Tensor,shape:tuple,s:int,n_probe:int=512,generator:torch.Generator=None)->float:
    '''
    median pairwise distance between random candidate patches at scale s; sets the per-scale smoothing
    '''
    # probe a random subset of images first, so a large (possibly CPU, float16) bank is never patchified whole
    idx=torch.randperm(X.shape[0],generator=generator,device=generator.device if generator is not None else "cpu")[:n_probe]
    patches,_=patch_bank(X[idx.to(X.device)].float(),shape,s)
    probe=patches[torch.randperm(patches.shape[0],generator=generator,device=generator.device if generator is not None else "cpu").to(patches.device)[:n_probe]]
    pair=torch.cdist(probe,probe)
    return pair[pair>0].median().item()


def patch_initial_sample(n:int,bank:dict,start_t:float,w:float=1.0,generator:torch.Generator=None)->torch.Tensor:
    '''
    exact sample from the patch mixture of one scale at start_t: every region picks its own patch
    (by the parent prior), so regions don't all start from the same whole image
    '''
    s,shape,patches=bank["s"],bank["shape"],bank["patches"]
    R=s*s
    N,P=patches.shape[0],patches.shape[-1]
    if bank["log_prior"] is None or w==0:
        probs=torch.full((N,),1.0/N,device=patches.device)
    else:
        probs=torch.softmax(w*bank["log_prior"].float(),dim=-1)
    idx=torch.multinomial(probs,n*R,replacement=True,generator=generator).reshape(n,R)
    if bank["positions"]=="same":
        chosen=patches[idx,torch.arange(R,device=patches.device).expand(n,-1)]
    else:
        chosen=patches[idx]
    x=unpatchify(chosen.reshape(n,R,P),shape,s)
    noise=torch.randn(x.shape,generator=generator,device=x.device,dtype=x.dtype)
    return start_t*x+(1-start_t)*noise


@torch.no_grad()
def sample_multiscale(n:int,X:torch.Tensor,shape:tuple,scales:list=(1,2,4),log_prior:torch.Tensor=None,w:float=1.0,
                      sigmas:list=None,M:int=1,steps:int=50,start_t:float=0.0,schedule:str="coarse_to_fine",
                      constant_weights:list=None,positions:str="any",patch_prior:str="text",switch_frac:float=0.5,
                      generator:torch.Generator=None,return_trajectory:bool=False):
    '''
    same Euler scheme as sample(), with k_t replaced by the scale-blended patch k_t
    X: (N,D) bank images (e.g. the top-k CLIP matches), shape: (C,H,W), sigmas: absolute smoothing per scale
    positions: "any" lets a region draw patches from every grid position, "same" only from its own
    the start at start_t is drawn from whichever scale has the most weight then: a whole image picked by
    the CLIP prior for scale 1, otherwise an independent prior-weighted patch per region
    patch_prior: "text" gives each candidate patch its parent's text prior, "none" only the whole-image scale
    (patches are then matched by visual similarity alone); switch_frac: see scale_weights
    '''
    assert 0<=start_t<1,"start_t must be in [0,1)"
    assert patch_prior in ["text","none"],f"unknown patch_prior {patch_prior}"
    banks=[]
    for s in scales:
        prior=log_prior if s==1 or patch_prior=="text" else None
        if positions=="same":
            patches=patchify(X,shape,s)
            bank_prior=None if prior is None else torch.log_softmax(prior,dim=-1)
        else:
            patches,bank_prior=patch_bank(X,shape,s,prior)
        banks.append({"s":s,"shape":shape,"positions":positions,"patches":patches,"log_prior":bank_prior,"x_sq":patches.pow(2).sum(-1)})
    start_alphas=scale_weights(start_t,len(scales),schedule,start_t,constant_weights,switch_frac)
    start_bank=banks[max(range(len(scales)),key=lambda j:start_alphas[j])]
    if start_bank["s"]==1:
        z=initial_sample(n,X,start_t,log_prior,w,generator)
    else:
        z=patch_initial_sample(n,start_bank,start_t,w,generator)
    ts=torch.linspace(start_t,1.0,steps+1).tolist()
    if ts[0]==0:
        ts[0]=1e-4
    trajectory=[z]
    for t,t_next in zip(ts[:-1],ts[1:]):
        alphas=scale_weights(t,len(scales),schedule,start_t,constant_weights,switch_frac)
        k=multiscale_k_t(z,banks,t,alphas,w,sigmas,M,generator)
        if t_next>=1.0:
            z=k/t
        else:
            z=z+(t_next-t)*(k-t*z)/(t*(1-t))
        if return_trajectory:
            trajectory.append(z)
    if return_trajectory:
        return z,trajectory
    return z
