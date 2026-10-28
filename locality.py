'''
Wiener-mask analytical denoiser (Lukoianov et al. 2025, "Locality in Image Diffusion Models Emerges from
Data Statistics", Algorithm 1), applied to the images of one tree node, in this repo's convention
z_t = a x + b eps with a = t, b = 1 - t (sigma_t^2 = b^2 / a^2).

with the node covariance S = U diag(lam) U^T:
    W_t = U diag(lam / (lam + sigma_t^2)) U^T                         (Wiener matrix, eq. 6)
    M_t = binarize(W_t, tau): M[q,j] = 1 if |W[q,j]| >= tau * max_j |W[q,j]|
    x0_hat^q = sum_i softmax_i( -sum_j M[q,j] (z_j - a x_i,j)^2 / (2 b^2) + w log p(i|c) ) x_i,q   (eq. 8)
so each output pixel q compares z to every image only on the pixels in its mask row. The paper suggests
(A.4) computing the statistics on the neighbors of the image being denoised instead of the whole dataset;
here the neighbors are the members of the tree node reached by the CLIP descent.
the linear Wiener filter (their baseline) is x0_hat = mu + W_t (z / a - mu).
'''
import math
import torch


def node_eig(Xn:torch.Tensor,S_global:torch.Tensor=None,shrinkage:float=0.0,ridge:float=0.0,lowrank:bool=None):
    '''
    mean and eigendecomposition of the node covariance, optionally blended toward a global covariance,
    S = (1 - shrinkage) S_node + shrinkage S_global, plus ridge * v * I, where v is the mean variance of
    S_global when shrinking (so the global part is shared by every node, see ShrunkNode) and of S_node otherwise.
    A node with n images has rank <= n - 1. At low noise the Wiener matrix tends to the projection onto the
    covariance's span, which is only the identity (and the masks only become local) if S is full rank,
    so small nodes need shrinkage toward a full-rank global covariance or a ridge
    returns (mean (d,), evals (k,), evecs (d,k), floor): S = evecs diag(evals) evecs^T + floor * (I - evecs evecs^T).
    without shrinkage and with n <= d (lowrank=None picks this automatically) this comes from an SVD of the
    n x d node instead of a d x d eigendecomposition, which is what makes pixel space (d = 3*H*W) tractable
    '''
    n,d=Xn.shape
    mean=Xn.mean(0)
    Xc=Xn-mean
    if lowrank is None:
        lowrank=shrinkage==0 and n<=d
    if lowrank:
        assert shrinkage==0,"the low-rank path can't blend in a full global covariance"
        _,svals,Vh=torch.linalg.svd(Xc,full_matrices=False)
        evals=svals.pow(2)/max(n-1,1)
        floor=ridge*evals.sum().item()/d
        return mean,evals+floor,Vh.T,floor
    S=Xc.T@Xc/max(n-1,1)
    v=S.diagonal().mean()
    if shrinkage>0:
        assert S_global is not None,"shrinkage needs S_global"
        S=(1-shrinkage)*S+shrinkage*S_global
        v=S_global.diagonal().mean()
    if ridge>0:
        S=S+ridge*v*torch.eye(d,device=S.device,dtype=S.dtype)
    evals,evecs=torch.linalg.eigh(S)
    return mean,evals.clamp_min(0),evecs,0.0


def global_spectrum(S_global:torch.Tensor):
    '''
    eigendecomposition of the global covariance, computed once per run for the Woodbury path
    '''
    lam,Q=torch.linalg.eigh(S_global)
    return lam.clamp_min(0),Q


class ShrunkNode:
    '''
    Wiener filter for S = shrinkage * S_global + (1 - shrinkage) S_node + rho I without a per-node eigendecomposition.
    write M = S + sigma2 I = A + U C U^T with
        A = Q diag(shrinkage * lam + rho + sigma2) Q^T   (global part, same for every node; Q, lam from global_spectrum)
        U = Xc^T (d x n), C = c I with c = (1 - shrinkage)/(n - 1)  (node part, rank <= n - 1)
    Woodbury: M^-1 = A^-1 - B K^-1 B^T with B = A^-1 U, K = I/c + U^T B (n x n), and W = S M^-1 = I - sigma2 M^-1:
        W = W_global + sigma2 B K^-1 B^T,   W_global = Q diag(1 - sigma2/(shrinkage*lam + rho + sigma2)) Q^T
    exact (same W as node_eig with the same shrinkage and ridge); per node per step it costs ~d^2 n instead of d^3
    Woodbury only pays off when the node has fewer images than dimensions (it solves an n x n system). A node with
    n > d (e.g. depth 0, the whole bank) instead eigendecomposes its d x d blended covariance once, which gives the
    same W; build the node once and reuse it (tree_sample caches nodes) since that decomposition is the expensive part
    '''
    def __init__(self,Xn:torch.Tensor,lam:torch.Tensor,Q:torch.Tensor,shrinkage:float,ridge:float=0.0):
        n,d=Xn.shape
        self.mean=Xn.mean(0)
        self.rho=ridge*lam.sum().item()/d
        self.exact=None
        if n>d:
            Xc=Xn-self.mean
            S=(1-shrinkage)*(Xc.T@Xc)/(n-1)+shrinkage*(Q*lam)@Q.T
            S.diagonal().add_(self.rho)
            evals,evecs=torch.linalg.eigh(S)
            self.exact=(evals.clamp_min(0),evecs)
            return
        self.U=(Xn-self.mean).T
        self.c=(1-shrinkage)/(n-1) if n>1 else 0.0
        self.lam,self.Q,self.shrinkage=lam,Q,shrinkage

    def global_diag(self,sigma2:float)->torch.Tensor:
        return self.shrinkage*self.lam+self.rho+sigma2

    def factors(self,sigma2:float):
        dvec=self.global_diag(sigma2)
        if self.c==0:
            return dvec,None,None
        B=self.Q@((self.Q.T@self.U)/dvec[:,None])
        K=torch.eye(self.U.shape[1],dtype=torch.float64,device=B.device)/self.c+(self.U.T@B).double()
        return dvec,B,torch.linalg.inv(K).to(B.dtype)

    def wiener_matrix(self,sigma2:float,W_global:torch.Tensor=None)->torch.Tensor:
        if self.exact is not None:
            return wiener_matrix(*self.exact,sigma2)
        dvec,B,Kinv=self.factors(sigma2)
        if W_global is None:
            W_global=(self.Q*(1-sigma2/dvec))@self.Q.T
        if B is None:
            return W_global
        return W_global+sigma2*(B@Kinv)@B.T

    def wiener_apply(self,r:torch.Tensor,sigma2:float)->torch.Tensor:
        if self.exact is not None:
            evals,evecs=self.exact
            return ((r@evecs)*(evals/(evals+sigma2)))@evecs.T
        '''
        W r for rows r (m,d), with no d x d matrix
        '''
        dvec,B,Kinv=self.factors(sigma2)
        Minv_r=((r@self.Q)/dvec)@self.Q.T
        if B is not None:
            Minv_r=Minv_r-((r@B)@Kinv.T)@B.T
        return r-sigma2*Minv_r


def wiener_matrix(evals:torch.Tensor,evecs:torch.Tensor,sigma2:float,floor:float=0.0)->torch.Tensor:
    '''
    W = evecs diag(evals/(evals+sigma2)) evecs^T + floor/(floor+sigma2) (I - evecs evecs^T)
    '''
    f_floor=floor/(floor+sigma2) if floor>0 else 0.0
    W=(evecs*(evals/(evals+sigma2)-f_floor))@evecs.T
    if f_floor>0:
        W=W+f_floor*torch.eye(W.shape[0],device=W.device,dtype=W.dtype)
    return W


def binarize(W:torch.Tensor,tau:float)->torch.Tensor:
    A=W.abs()
    return (A>=tau*A.amax(1,keepdim=True)).to(W.dtype)


def masked_denoise(z:torch.Tensor,Xn:torch.Tensor,t:float,mask:torch.Tensor,log_prior:torch.Tensor=None,w:float=0.0,
                   chunk:int=256,max_elems:float=1.5e8)->torch.Tensor:
    '''
    eq. 8 with a per-pixel binary mask, streaming the softmax over chunks of images
    z: (B,d), Xn: (n,d), mask: (d,d) with row q = pixels that output pixel q looks at, log_prior: (n,)
    '''
    a,b=t,1-t
    B,d=z.shape
    # keep the (B, chunk, d) temporaries around max_elems entries so large batches fit in GPU memory
    chunk=max(1,min(chunk,int(max_elems//(B*d))))
    m=torch.full((B,d),-math.inf,device=z.device,dtype=z.dtype)
    s=torch.zeros(B,d,device=z.device,dtype=z.dtype)
    acc=torch.zeros(B,d,device=z.device,dtype=z.dtype)
    for i in range(0,Xn.shape[0],chunk):
        Xc=Xn[i:i+chunk]
        D=(z[:,None,:]-a*Xc[None]).pow(2)                     # (B,c,d) per-pixel squared distances
        logits=-(D@mask.T)/(2*b*b)                              # (B,c,d): masked distance for each output pixel
        if log_prior is not None and w!=0:
            logits=logits+w*log_prior[i:i+chunk][None,:,None]
        m_new=torch.maximum(m,logits.amax(1))
        scale=torch.exp(m-m_new)
        e=torch.exp(logits-m_new[:,None])
        s=s*scale+e.sum(1)
        acc=acc*scale+(e*Xc[None]).sum(1)
        m=m_new
    return acc/s


def wiener_denoise(z:torch.Tensor,t:float,mean:torch.Tensor,evals:torch.Tensor,evecs:torch.Tensor,floor:float=0.0)->torch.Tensor:
    '''
    linear Wiener filter: posterior mean if the node were Gaussian N(mean, S); applied in low-rank form so
    no d x d matrix is built
    '''
    a,b=t,1-t
    sigma2=(b/a)**2
    f_floor=floor/(floor+sigma2) if floor>0 else 0.0
    r=z/a-mean
    return mean+f_floor*r+((r@evecs)*(evals/(evals+sigma2)-f_floor))@evecs.T


@torch.no_grad()
def locality_sample(n:int,Xn:torch.Tensor,stats:tuple,kind:str="locality",tau:float=0.02,steps:int=10,t_min:float=1e-3,
                    log_prior:torch.Tensor=None,w:float=0.0,generator:torch.Generator=None,chunk:int=256,
                    z0:torch.Tensor=None)->torch.Tensor:
    '''
    Euler on v = (x0_hat - z)/(1 - t) from pure noise at t_min to t = 1 (the deterministic DDIM update for
    this interpolation); the last step lands on x0_hat
    stats: (mean, evals, evecs, floor) from node_eig; kind: "locality" (masked, eq. 8) or "wiener" (linear filter)
    '''
    mean,evals,evecs,floor=stats
    z=torch.randn((n,Xn.shape[1]),generator=generator,device=Xn.device,dtype=Xn.dtype) if z0 is None else z0
    ts=torch.linspace(t_min,1.0,steps+1).tolist()
    for t,t_next in zip(ts[:-1],ts[1:]):
        if kind=="wiener":
            x0=wiener_denoise(z,t,mean,evals,evecs,floor)
        else:
            mask=binarize(wiener_matrix(evals,evecs,((1-t)/t)**2,floor),tau)
            x0=masked_denoise(z,Xn,t,mask,log_prior,w,chunk)
        z=z+(t_next-t)*(x0-z)/(1-t)
    return z


@torch.no_grad()
def locality_sample_shrunk(groups:list,lam:torch.Tensor,Q:torch.Tensor,shrinkage:float,ridge:float=0.0,kind:str="locality",
                           tau:float=0.02,steps:int=10,t_min:float=1e-3,w:float=0.0,generator:torch.Generator=None,
                           chunk:int=256,z0s:list=None,nodes:list=None)->list:
    '''
    locality_sample for several nodes at once with shrinkage toward the global covariance, via ShrunkNode.
    all nodes step through the same time grid together, so the shared global part of the Wiener matrix is
    built once per step (one d x d x d matmul) instead of one eigendecomposition per node
    groups: list of (Xn, n_samples, log_prior or None); returns a list of (n_samples, d) tensors
    z0s: optional list of starting noise per group (default: drawn from generator)
    nodes: optional prebuilt ShrunkNode per group (reused across calls; built here otherwise)
    '''
    if not groups:
        return []
    if nodes is None:
        nodes=[ShrunkNode(Xn,lam,Q,shrinkage,ridge) for Xn,_,_ in groups]
    rho=ridge*lam.sum().item()/lam.shape[0]
    zs=list(z0s) if z0s is not None else [torch.randn((n,Xn.shape[1]),generator=generator,device=Xn.device,dtype=Xn.dtype) for Xn,n,_ in groups]
    ts=torch.linspace(t_min,1.0,steps+1).tolist()
    for t,t_next in zip(ts[:-1],ts[1:]):
        sigma2=((1-t)/t)**2
        W_global=None
        if kind=="locality" and any(node.exact is None for node in nodes):
            # shared by the Woodbury nodes; nodes with more images than dimensions use their own exact W
            dvec=shrinkage*lam+rho+sigma2
            W_global=(Q*(1-sigma2/dvec))@Q.T
        for j,((Xn,_,log_prior),node) in enumerate(zip(groups,nodes)):
            z=zs[j]
            if kind=="wiener":
                x0=node.mean+node.wiener_apply(z/t-node.mean,sigma2)
            else:
                mask=binarize(node.wiener_matrix(sigma2,W_global),tau)
                x0=masked_denoise(z,Xn,t,mask,log_prior,w,chunk)
            zs[j]=z+(t_next-t)*(x0-z)/(1-t)
    return zs


@torch.no_grad()
def low_rank_cov(X:torch.Tensor,k:int,device,oversample:int=16,iters:int=2,chunk:int=2048,generator:torch.Generator=None):
    '''
    top-k principal components of the rows of X (N,d) by randomized SVD with power iterations, streaming row
    chunks to device, so X can be a large float16 tensor in CPU memory (e.g. a 256px pixel bank or one node of it)
    returns mean (d,), evals (k',) covariance eigenvalues, V (d,k') orthonormal, resid: the average variance in
    the directions left out, (trace(S) - sum(evals)) / (d - k'), so mean diag and trace of the covariance are kept
    '''
    N,d=X.shape
    def rows(i):
        return X[i:i+chunk].to(device).float()
    mean=torch.zeros(d,device=device)
    for i in range(0,N,chunk):
        mean+=rows(i).sum(0)
    mean/=N
    total=0.0
    for i in range(0,N,chunk):
        total+=(rows(i)-mean).pow(2).sum().item()
    trace=total/max(N-1,1)
    l=min(k+oversample,N,d)
    def times(M):
        # Xc @ M for M (d,l)
        return torch.cat([(rows(i)-mean)@M for i in range(0,N,chunk)])
    def times_t(Y):
        # Xc^T @ Y for Y (N,l)
        out=torch.zeros(d,Y.shape[1],device=device)
        for i in range(0,N,chunk):
            out+=(rows(i)-mean).T@Y[i:i+chunk]
        return out
    omega=torch.randn(d,l,device=device,generator=generator)
    Y=torch.linalg.qr(times(omega)).Q
    for _ in range(iters):
        Y=torch.linalg.qr(times(torch.linalg.qr(times_t(Y)).Q)).Q
    B=times_t(Y).T                                   # (l,d) = Y^T Xc
    _,s,Vh=torch.linalg.svd(B,full_matrices=False)
    k=min(k,l)
    evals=s[:k].pow(2)/max(N-1,1)
    keep=evals>evals.max().clamp_min(1e-30)*1e-10
    evals,V=evals[keep],Vh[:k][keep].T
    resid=max(trace-evals.sum().item(),0.0)/max(d-V.shape[1],1)
    return mean,evals,V,resid


class LowRankWiener:
    '''
    Wiener filter for a covariance S = c0 I + U diag(c) U^T (U is d x m, any columns), without d x d matrices:
    with M = S + sigma2 I = alpha I + U diag(c) U^T, alpha = c0 + sigma2, Woodbury gives
        M^-1 r = (r - U (alpha diag(1/c) + U^T U)^-1 U^T r) / alpha,   W r = r - sigma2 M^-1 r
    built from low_rank_cov factors of the node (and optionally the global bank), mirroring node_eig:
        S = (1 - shrinkage) S_node + shrinkage S_global + ridge * v I
    where each low-rank factor keeps its left-out variance as an isotropic term, and v is the mean variance
    of S_global when shrinking and of S_node otherwise
    '''
    def __init__(self,node:tuple,global_factors:tuple=None,shrinkage:float=0.0,ridge:float=0.0):
        mean,evals_n,V_n,resid_n=node
        self.mean=mean
        Us=[V_n]
        cs=[(1-shrinkage)*evals_n] if shrinkage<1 else [torch.zeros_like(evals_n)]
        c0=(1-shrinkage)*resid_n
        v=(evals_n.sum().item()+resid_n*(V_n.shape[0]-V_n.shape[1]))/V_n.shape[0]
        if shrinkage>0:
            assert global_factors is not None,"shrinkage needs the global low-rank factors"
            _,evals_g,V_g,resid_g=global_factors
            Us.append(V_g)
            cs.append(shrinkage*evals_g)
            c0+=shrinkage*resid_g
            v=(evals_g.sum().item()+resid_g*(V_g.shape[0]-V_g.shape[1]))/V_g.shape[0]
        U=torch.cat(Us,1)
        c=torch.cat(cs)
        keep=c>0
        self.U,self.c=U[:,keep],c[keep]
        self.c0=c0+ridge*v
        self.UtU=(self.U.T@self.U).double()

    def apply(self,r:torch.Tensor,sigma2:float)->torch.Tensor:
        alpha=self.c0+sigma2
        Ur=(r@self.U).double()                                          # (B,m)
        K=alpha*torch.diag(1/self.c.double())+self.UtU
        y=torch.linalg.solve(K,Ur.T).T.to(r.dtype)                     # (B,m)
        Minv_r=(r-y@self.U.T)/alpha
        return r-sigma2*Minv_r


@torch.no_grad()
def wiener_lowrank_sample(n:int,stats:LowRankWiener,steps:int=10,t_min:float=1e-3,generator:torch.Generator=None,
                          z0:torch.Tensor=None)->torch.Tensor:
    '''
    the linear Wiener sampler (as locality_sample(kind="wiener")) with LowRankWiener statistics
    '''
    d=stats.mean.shape[0]
    z=torch.randn((n,d),generator=generator,device=stats.mean.device) if z0 is None else z0
    ts=torch.linspace(t_min,1.0,steps+1).tolist()
    for t,t_next in zip(ts[:-1],ts[1:]):
        x0=stats.mean+stats.apply(z/t-stats.mean,((1-t)/t)**2)
        z=z+(t_next-t)*(x0-z)/(1-t)
    return z
