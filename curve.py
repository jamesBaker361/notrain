'''
curve node sampler: a 1-D parametric curve through a node's images, and samples from along it.

1. order: the images' order along the curve is an open traveling-salesman path (shortest path visiting every image
   once). tsp_path uses LKH (Helsgaun's Lin-Kernighan, via the optional elkai package) if installed, else chained
   2-opt: 2-opt local search plus double-bridge kicks that are kept when they shorten the path, the same
   kick-and-reoptimize chaining as Chained Lin-Kernighan with a 2-opt local search instead of LK moves.
   An open path is a closed tour through one extra "dummy" city at distance 0 from every image, cut at the dummy.
2. parameter: t_i = cumulative path length up to image i, scaled to [0,1].
3. fit: x(t) = B(t) A with a B-spline basis B (n_knots functions, knots at quantiles of the images' t) and a
   second-difference penalty on the coefficients A (a P-spline). The default, degree 1 with one knot per image and
   no smoothing, is exactly the polyline through the images in path order: samples are the images and crossfades
   between path neighbors. Cubic curves (degree 3) are smoother but overshoot between dissimilar images, which in
   latent space means out-of-distribution latents (saturated, noisy decodes); fewer knots average more images.
4. sample: mode "random" draws t ~ p(t) on a fine grid and returns x(t), the point every sample is brought to;
   p(t) is uniform in t, or with text exp(w * log p(t | c)), where log p(. | c) is the members' CLIP log prior
   fitted along the curve with the same basis. Mode "flow" instead runs the probability-flow ODE of the uniform
   (text-weighted) distribution on the curve: the ideal denoiser over the grid points, like CFDM over images,
   so each sample lands on the curve point its noise flows to.
'''
import math
import numpy as np
import torch


def path_length(path:np.ndarray,D:np.ndarray)->float:
    return float(D[path[:-1],path[1:]].sum())


def two_opt(tour:np.ndarray,D:np.ndarray)->np.ndarray:
    '''
    2-opt local search on a closed tour (array of city ids, tour[-1] connects back to tour[0]): for each edge
    (a,b) take the best exchange with an edge (c,d), reversing b..c, until no exchange shortens the tour
    '''
    tour=tour.copy()
    N=len(tour)
    improved=True
    while improved:
        improved=False
        for i in range(N-2):
            a,b=tour[i],tour[i+1]
            c=tour[i+2:]
            d=np.append(tour[i+3:],tour[0])
            delta=D[a,c]+D[b,d]-D[a,b]-D[c,d]
            if i==0:
                # (c,d)=(tour[-1],tour[0]) shares the city a
                delta[-1]=0
            j=int(np.argmin(delta))
            if delta[j]<-1e-9:
                tour[i+1:i+3+j]=tour[i+1:i+3+j][::-1]
                improved=True
    return tour


def double_bridge(tour:np.ndarray,rng:np.random.Generator)->np.ndarray:
    # the standard Lin-Kernighan kick: cut into A B C D, reconnect as A C B D (not undoable by one 2-opt move)
    i,j,k=np.sort(rng.choice(np.arange(1,len(tour)),3,replace=False))
    return np.concatenate([tour[:i],tour[j:k],tour[i:j],tour[k:]])


def chained_two_opt(D:np.ndarray,kicks:int=30,seed:int=0)->np.ndarray:
    '''
    closed tour over the cities of D: nearest-neighbor start, 2-opt, then `kicks` double-bridge kicks, each followed
    by 2-opt and kept only if the tour got shorter
    '''
    N=len(D)
    rng=np.random.default_rng(seed)
    tour=[0]
    left=np.ones(N,dtype=bool)
    left[0]=False
    for _ in range(N-1):
        dist=np.where(left,D[tour[-1]],np.inf)
        nxt=int(np.argmin(dist))
        tour.append(nxt)
        left[nxt]=False
    tour=two_opt(np.array(tour),D)
    closed=lambda t:path_length(np.append(t,t[0]),D)
    best=closed(tour)
    if N<8:
        return tour
    for _ in range(kicks):
        cand=two_opt(double_bridge(tour,rng),D)
        length=closed(cand)
        if length<best-1e-9:
            tour,best=cand,length
    return tour


def tsp_path(D:np.ndarray,method:str="auto",kicks:int=30,seed:int=0)->np.ndarray:
    '''
    shortest open path visiting every point once (approximately), as an order of the indices of D (n x n)
    method: "2opt" (chained 2-opt), "lkh" (elkai/LKH, must be installed), "auto" (LKH if available)
    '''
    n=len(D)
    if n<=2:
        return np.arange(n)
    # dummy city n at distance 0 from everything: an optimal closed tour through it is an optimal open path
    Dd=np.zeros((n+1,n+1))
    Dd[:n,:n]=D
    tour=None
    if method in ["auto","lkh"]:
        try:
            import elkai
            scale=1e6/max(D.max(),1e-12)
            tour=np.array(elkai.DistanceMatrix(np.round(Dd*scale).astype(int).tolist()).solve_tsp())[:-1]
        except ImportError:
            if method=="lkh":
                raise
    if tour is None:
        tour=chained_two_opt(Dd,kicks,seed)
    cut=int(np.flatnonzero(tour==n)[0])
    return np.concatenate([tour[cut+1:],tour[:cut]])


def spline_knots(t:np.ndarray,n_knots:int,degree:int)->tuple:
    '''
    clamped knot vector for n_knots basis functions with inner knots at quantiles of the data parameters t, so
    every segment holds about as many images (n_knots = len(t), degree 1 puts a knot at every image)
    '''
    k=min(degree,n_knots-1)
    inner=np.quantile(t,np.linspace(0,1,n_knots-k+1))
    inner[0],inner[-1]=0.0,1.0
    # strictly increasing (duplicate t values would make empty segments)
    inner=np.maximum.accumulate(inner+np.arange(len(inner))*1e-9)
    inner=inner/inner[-1]
    return np.concatenate([[0]*k,inner,[1]*k]),k


def bspline_basis(t:np.ndarray,knots:np.ndarray,k:int)->np.ndarray:
    # (len(t), n_knots) B-spline design matrix on [0,1]
    from scipy.interpolate import BSpline
    return BSpline.design_matrix(np.clip(t,0,1),knots,k).toarray()


class Curve:
    '''
    a P-spline curve through the rows of Xn (m, d) in TSP order. Holds the basis on a grid of `grid` parameter
    values; fit() maps any per-image values (the images themselves, a log prior) to values along the grid
    n_knots 0: one per image; grid 0: 16 points per image, at least 1024, at most max_grid_bytes of curve points
    '''
    def __init__(self,Xn:torch.Tensor,n_knots:int=0,smooth:float=0.0,grid:int=0,tsp:str="auto",kicks:int=30,seed:int=0,
                 degree:int=1,max_grid_bytes:float=2e9):
        m=Xn.shape[0]
        D=torch.cdist(Xn,Xn).cpu().double().numpy()
        self.order=torch.as_tensor(tsp_path(D,tsp,kicks,seed),dtype=torch.long)
        steps=D[self.order[:-1].numpy(),self.order[1:].numpy()]
        total=steps.sum()
        self.length=float(total)
        t=np.concatenate([[0],np.cumsum(steps)])/max(total,1e-12)
        self.t=torch.as_tensor(t,dtype=torch.float32)
        K=max(2,min(n_knots or m,m))
        knots,k=spline_knots(t,K,degree)
        B=bspline_basis(t,knots,k)
        # second differences of the coefficients; lam is relative to the data term so `smooth` is scale-free
        Dk=np.diff(np.eye(K),2,axis=0) if K>2 else np.zeros((0,K))
        BtB=B.T@B
        lam=smooth*np.trace(BtB)/max(np.trace(Dk.T@Dk),1e-12)
        # tiny ridge keeps the solve well-posed when some basis functions see no images
        self.L=torch.as_tensor(np.linalg.solve(BtB+lam*Dk.T@Dk+1e-8*np.eye(K),B.T),dtype=torch.float32)
        if not grid:
            grid=int(max(256,min(max(1024,16*m),max_grid_bytes//(4*Xn.shape[1]))))
        self.grid_t=torch.linspace(0,1,grid)
        self.Bg=torch.as_tensor(bspline_basis(self.grid_t.numpy(),knots,k),dtype=torch.float32)
        self.points=self.fit(Xn)

    def fit(self,values:torch.Tensor)->torch.Tensor:
        '''
        values: (m, ...) per image in the node's row order -> (grid, ...) along the curve
        '''
        dev=values.device
        ordered=values[self.order.to(dev)]
        flat=ordered.reshape(len(ordered),-1).float()
        return (self.Bg.to(dev)@(self.L.to(dev)@flat)).reshape(len(self.grid_t),*values.shape[1:])


def curve_sample(n:int,curve:Curve,log_prior:torch.Tensor=None,w:float=0.0,mode:str="random",steps:int=10,
                 generator:torch.Generator=None)->torch.Tensor:
    '''
    n samples on the curve. log_prior: (m,) log p(image | text) of the node's members (row order), fitted along
    the curve and weighted by w; mode "random" picks t ~ p(t) directly, "flow" runs the ideal-denoiser flow
    toward the curve's grid points from Gaussian noise (z_t = t x + (1-t) eps, Euler from t=0 to 1)
    '''
    C=curve.points
    device=C.device
    logp=torch.zeros(len(C),device=device)
    if log_prior is not None and w!=0:
        logp=w*curve.fit(log_prior.to(device).float())
    if mode=="random":
        # inverse CDF on the grid (works with a generator on any device)
        cdf=torch.cumsum(torch.softmax(logp,0),0)
        u=torch.rand(n,generator=generator,device=device)*cdf[-1]
        idx=torch.searchsorted(cdf,u).clamp_max(len(C)-1)
        return C[idx].clone()
    assert mode=="flow",f"unknown curve mode {mode}"
    z=torch.randn((n,C.shape[1]),generator=generator,device=device)
    c_sq=C.pow(2).sum(-1)
    ts=torch.linspace(0,1,steps+1).tolist()
    for t0,t1 in zip(ts[:-1],ts[1:]):
        # posterior over grid points: log N(z; t c, (1-t)^2 I) + text log prior
        sq=(z.pow(2).sum(-1,keepdim=True)-2*t0*z@C.T+t0**2*c_sq)
        weights=torch.softmax(-sq/(2*(1-t0)**2)+logp,dim=-1)
        x0=weights@C
        z=z+(t1-t0)*(x0-z)/(1-t0)
    return z
