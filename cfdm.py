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
           steps:int=50,start_t:float=0.0,generator:torch.Generator=None,return_trajectory:bool=False):
    '''
    forward Euler from start_t to t=1 (Algorithm 1). The last step from t to 1 lands exactly
    on smoothed_k_t(z)/t, so nothing blows up at t=1.
    X: (N,D) training points (flattened pixels or latents), log_prior: (N,) from text_log_prior
    '''
    assert 0<=start_t<1,"start_t must be in [0,1)"
    x_sq=X.pow(2).sum(-1)
    z=initial_sample(n,X,start_t,log_prior,w,generator)
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
