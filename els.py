'''
closed-form denoisers in x0-prediction form, for comparing against a trained network
(Kamb & Ganguli, "An analytic theory of creativity in convolutional diffusion models", plus a text prior)

convention as in cfdm.py: z_t = t x + (1-t) eps, so E[x|z_t] is a softmax-weighted average with logits
    -||z - t x||^2 / (2 (1-t)^2) + w * log p(i|c)
and the probability-flow velocity is v = (x0_hat - z) / (1-t).

ideal_denoise: whole-image posterior mean (the exact optimal denoiser, memorizes)
ls_denoise:    local score machine. Each output pixel is the posterior mean of the center pixel of a
               P x P patch, over the patches of every training image at the same position
els_denoise:   equivariant local score machine. Same, but candidates are P x P patches from every
               position of every (candidate) training image
patches are zero-padded at the borders, like a conv net with zero padding.
X: (N,C,H,W) training latents, z: (B,C,H,W), log_prior: (N,) or (B,N) over training images.
'''
import math
import torch
import torch.nn.functional as F


def text_log_prior_from(text_embeds:torch.Tensor,keys:torch.Tensor,beta:float)->torch.Tensor:
    '''
    log p(i|c) = log_softmax_i(beta * cos(c, key_i)); keys are CLIP image embeddings T (image prior)
    or CLIP text embeddings of the training captions C (caption prior)
    '''
    return torch.log_softmax(beta*text_embeds@keys.T,dim=-1)


def _prior_logits(log_prior,w,B,N,device):
    if log_prior is None or w==0:
        return torch.zeros(B,N,device=device)
    lp=log_prior if log_prior.dim()==2 else log_prior[None].expand(B,-1)
    return w*lp


def ideal_denoise(z:torch.Tensor,X:torch.Tensor,t:float,log_prior:torch.Tensor=None,w:float=1.0)->torch.Tensor:
    B,N=z.shape[0],X.shape[0]
    zf,Xf=z.flatten(1),X.flatten(1)
    logits=(2*t*zf@Xf.T-t*t*Xf.pow(2).sum(-1))/(2*(1-t)**2)+_prior_logits(log_prior,w,B,N,z.device)
    return (torch.softmax(logits,dim=-1)@Xf).reshape(z.shape)


def _box_sum(x:torch.Tensor,P:int)->torch.Tensor:
    # sum over the P x P window centered on each pixel, zero padded; x: (...,H,W)
    shape=x.shape
    x=x.reshape(-1,1,*shape[-2:])
    out=F.avg_pool2d(x,P,stride=1,padding=P//2,count_include_pad=True)*(P*P)
    return out.reshape(shape)


def ls_denoise(z:torch.Tensor,X:torch.Tensor,t:float,P:int,log_prior:torch.Tensor=None,w:float=1.0,
               x_sq:torch.Tensor=None,chunk:int=4096)->torch.Tensor:
    '''
    local (same-position) patch posterior mean at every pixel. The patch distance is a box filter of
    per-pixel products, so no patches are materialized; the softmax over images streams over chunks of N
    x_sq: optional precomputed _box_sum(X.pow(2).sum(1), P) of shape (N,H,W)
    '''
    assert P%2==1,"patch size must be odd"
    B,C,H,W=z.shape
    N=X.shape[0]
    if x_sq is None:
        x_sq=_box_sum(X.pow(2).sum(1),P)
    prior=_prior_logits(log_prior,w,B,N,z.device)
    # streaming softmax: running max m, normalizer s, and weighted sum acc per (b,h,w)
    m=torch.full((B,H,W),-math.inf,device=z.device,dtype=z.dtype)
    s=torch.zeros(B,H,W,device=z.device,dtype=z.dtype)
    acc=torch.zeros(B,C,H,W,device=z.device,dtype=z.dtype)
    for i in range(0,N,chunk):
        Xc=X[i:i+chunk]
        cross=_box_sum(torch.einsum("bchw,nchw->bnhw",z,Xc),P)
        logits=(2*t*cross-t*t*x_sq[i:i+chunk][None])/(2*(1-t)**2)+prior[:,i:i+chunk,None,None]
        m_new=torch.maximum(m,logits.amax(1))
        scale=torch.exp(m-m_new)
        e=torch.exp(logits-m_new[:,None])
        s=s*scale+e.sum(1)
        acc=acc*scale[:,None]+torch.einsum("bnhw,nchw->bchw",e,Xc)
        m=m_new
    return acc/s[:,None]


def _unfold_patches(x:torch.Tensor,P:int)->torch.Tensor:
    # (B,C,H,W) -> (B,H*W,C*P*P), zero padded, one patch centered on each pixel
    return F.unfold(x,P,padding=P//2).transpose(1,2)


def els_denoise(z:torch.Tensor,X:torch.Tensor,t:float,P:int,log_prior:torch.Tensor=None,w:float=1.0,
                query_chunk:int=256,cand_chunk:int=65536)->torch.Tensor:
    '''
    equivariant patch posterior mean at every pixel: candidates are patches centered on every pixel of
    every image in X (pass a subset of images to keep this tractable: cost is B*H*W x N*H*W).
    each candidate patch carries its parent image's prior
    '''
    assert P%2==1,"patch size must be odd"
    B,C,H,W=z.shape
    N=X.shape[0]
    R=H*W
    cand=_unfold_patches(X,P).reshape(N*R,-1)
    centers=X.permute(0,2,3,1).reshape(N*R,C)
    c_sq=cand.pow(2).sum(-1)
    prior=_prior_logits(log_prior,w,B,N,z.device).repeat_interleave(R,dim=1)
    q_all=_unfold_patches(z,P)
    out=torch.empty(B,R,C,device=z.device,dtype=z.dtype)
    for b in range(B):
        for q0 in range(0,R,query_chunk):
            q=q_all[b,q0:q0+query_chunk]
            m=torch.full((q.shape[0],),-math.inf,device=z.device,dtype=z.dtype)
            s=torch.zeros(q.shape[0],device=z.device,dtype=z.dtype)
            acc=torch.zeros(q.shape[0],C,device=z.device,dtype=z.dtype)
            for c0 in range(0,N*R,cand_chunk):
                logits=(2*t*q@cand[c0:c0+cand_chunk].T-t*t*c_sq[c0:c0+cand_chunk])/(2*(1-t)**2)+prior[b,c0:c0+cand_chunk]
                m_new=torch.maximum(m,logits.amax(1))
                scale=torch.exp(m-m_new)
                e=torch.exp(logits-m_new[:,None])
                s=s*scale+e.sum(1)
                acc=acc*scale[:,None]+e@centers[c0:c0+cand_chunk]
                m=m_new
            out[b,q0:q0+query_chunk]=acc/s[:,None]
    return out.transpose(1,2).reshape(B,C,H,W)


def denoise(kind:str,z:torch.Tensor,X:torch.Tensor,t:float,P:int=None,log_prior:torch.Tensor=None,w:float=1.0,**kwargs)->torch.Tensor:
    if kind=="ideal":
        return ideal_denoise(z,X,t,log_prior,w)
    if kind=="ls":
        return ls_denoise(z,X,t,P,log_prior,w,**kwargs)
    if kind=="els":
        return els_denoise(z,X,t,P,log_prior,w,**kwargs)
    raise ValueError(f"unknown denoiser {kind}")


def r2(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    '''
    per-sample r^2 of pred against target: 1 - ||pred - target||^2 / ||target - mean(target)||^2
    '''
    pred,target=pred.flatten(1),target.flatten(1)
    resid=(pred-target).pow(2).sum(-1)
    total=(target-target.mean(-1,keepdim=True)).pow(2).sum(-1)
    return 1-resid/total


@torch.no_grad()
def machine_sample(kind:str,z0:torch.Tensor,X:torch.Tensor,patch_schedule,steps:int=50,log_prior:torch.Tensor=None,
                   w:float=1.0,t_min:float=1e-3,**kwargs)->torch.Tensor:
    '''
    Euler on v = (x0_hat - z)/(1-t) from t_min to 1 with a closed-form denoiser
    patch_schedule: callable t -> P (ignored for "ideal"); z0: (B,C,H,W) initial noise
    the last step lands exactly on x0_hat
    '''
    ts=torch.linspace(t_min,1.0,steps+1).tolist()
    z=z0
    for t,t_next in zip(ts[:-1],ts[1:]):
        x0=denoise(kind,z,X,t,patch_schedule(t) if kind!="ideal" else None,log_prior,w,**kwargs)
        z=z+(t_next-t)*(x0-z)/(1-t)
    return z
