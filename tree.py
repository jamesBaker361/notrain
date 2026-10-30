'''
hierarchical CLIP tree over a bank built by embed.py

build: PCA on the CLIP image embeddings T, then clustering in PCA space (--cluster_method): recursive k-means
(k fixed, or chosen by silhouette score), recursive DBSCAN, or one agglomerative dendrogram (binary tree).
A node with --c_build or fewer images gets one leaf child per image, so every leaf holds exactly one image. Members of every node are
contiguous in tree["perm"], so node i owns bank rows perm[start[i]:end[i]].
per node we store the mean CLIP embedding (full and PCA coords) and, for internal nodes, the mean and
per-dim variance of the members' VAE latents (a leaf's stats are just its latent with variance 0).

sample: descend from the root, picking a child with probability softmax(w * logit_scale * score), where
score is the child's mean CLIP similarity to the prompt (computed in PCA space; the constant c.mu term
cancels in the softmax). w=0 is a uniform random walk (unconditional), large w is greedy. Stop at depth
--tree_depth or at the first node with <= --c_search images, then sample from that node with
--node_sampler: diagonal Gaussian, low-rank Gaussian of the members, CFDM over the members, or the
Wiener-mask analytical model of Lukoianov et al. (locality.py) using the node's own covariance.

usage: sbatch runpygpu_chip.sh tree.py --bank banks/celeb.pt --save_path banks/celeb_tree.pt --branching silhouette
'''
import os
import argparse
import time
import math
import numpy as np
import torch

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celeb.pt")
parser.add_argument("--save_path",type=str,default="banks/celeb_tree.pt")
parser.add_argument("--pca_dim",type=int,default=64)
parser.add_argument("--branching",type=str,default="10",help="k for every k-means split, or 'silhouette' to pick k per node")
parser.add_argument("--k_min",type=int,default=2,help="smallest k tried with --branching silhouette")
parser.add_argument("--k_max",type=int,default=10,help="largest k tried with --branching silhouette")
parser.add_argument("--silhouette_sample",type=int,default=2000,help="points used to estimate the silhouette score")
parser.add_argument("--c_build",type=int,default=10,help="nodes with this many images or fewer get one leaf per image")
parser.add_argument("--n_init",type=int,default=3,help="k-means restarts")
parser.add_argument("--cluster_method",type=str,default="kmeans",choices=["kmeans","dbscan","agglomerative"],help="how nodes are split: recursive k-means, recursive DBSCAN, or one agglomerative dendrogram")
parser.add_argument("--dbscan_min_samples",type=int,default=5,help="dbscan: min_samples, and the neighbor whose distance sets eps")
parser.add_argument("--dbscan_quantile",type=float,default=0.5,help="dbscan: eps = this quantile of each node's min_samples-th-neighbor distances")
parser.add_argument("--dbscan_noise",type=str,default="nearest",choices=["nearest","group"],help="dbscan: noise points join their nearest cluster, or form one extra child")
parser.add_argument("--vae_stats",type=str,default="auto",choices=["auto","yes","no"],help="store per-node mean/variance of the bank vectors (diagonal sampler only); auto skips them when over 4GB")
parser.add_argument("--linkage",type=str,default="ward",choices=["ward","average","complete","single"],help="agglomerative: scipy linkage method")
parser.add_argument("--cluster_on",type=str,default="image",choices=["image","caption","mix"],help="embeddings the tree is built from (and the descent compares the prompt to): CLIP image embeddings T, CLIP text embeddings of the captions C, or a normalized blend")
parser.add_argument("--caption_weight",type=float,default=0.5,help="mix: weight a of the captions in normalize((1-a) T + a C)")
parser.add_argument("--seed",type=int,default=123)


def choose_k(Y:np.ndarray,branching:str,k_min:int,k_max:int,silhouette_sample:int,n_init:int,seed:int):
    '''
    returns k-means labels for Y, with k fixed or chosen by the best silhouette score
    '''
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n=Y.shape[0]
    if branching!="silhouette":
        k=min(int(branching),n)
        return KMeans(n_clusters=k,n_init=n_init,random_state=seed).fit_predict(Y)
    best_labels,best_score=None,-math.inf
    for k in range(k_min,min(k_max,n-1)+1):
        labels=KMeans(n_clusters=k,n_init=n_init,random_state=seed).fit_predict(Y)
        if len(np.unique(labels))<2:
            continue
        score=silhouette_score(Y,labels,sample_size=min(n,silhouette_sample),random_state=seed)
        if score>best_score:
            best_labels,best_score=labels,score
    if best_labels is None:
        best_labels=KMeans(n_clusters=min(k_min,n),n_init=n_init,random_state=seed).fit_predict(Y)
    return best_labels


def dbscan_groups(Y:np.ndarray,min_samples:int,quantile:float,seed:int,noise:str="nearest"):
    '''
    split one node with DBSCAN. eps is set per node from the distribution of distances to each point's
    min_samples-th neighbor (a fixed eps stops working as nodes get smaller and denser). Noise points (DBSCAN's
    sparse border points) join the cluster of their nearest clustered point (noise="nearest"), or become one
    extra group that is split further down (noise="group", which mixes border points of every cluster).
    Returns relative index groups, or None if no eps gives 2+ groups
    '''
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    n=Y.shape[0]
    k=max(1,min(min_samples,n-1))
    knn=NearestNeighbors(n_neighbors=k+1).fit(Y).kneighbors(Y)[0][:,-1]
    for q in [quantile,quantile/2,min(0.95,2*quantile),quantile/4]:
        eps=float(np.quantile(knn,q))
        if eps<=0:
            continue
        labels=DBSCAN(eps=eps,min_samples=k).fit_predict(Y)
        if (labels>=0).sum()==0:
            continue
        if noise=="nearest" and (labels<0).any():
            clustered=np.flatnonzero(labels>=0)
            nearest=NearestNeighbors(n_neighbors=1).fit(Y[clustered]).kneighbors(Y[labels<0])[1][:,0]
            labels[labels<0]=labels[clustered[nearest]]
        groups=[np.flatnonzero(labels==l) for l in np.unique(labels) if l>=0]
        if (labels<0).any():
            groups.append(np.flatnonzero(labels<0))
        if len(groups)>=2:
            return groups
    return None


def build_tree(T:torch.Tensor,pca_dim:int=64,branching:str="10",k_min:int=2,k_max:int=10,silhouette_sample:int=2000,
               c_build:int=10,n_init:int=3,seed:int=123,method:str="kmeans",dbscan_min_samples:int=5,
               dbscan_quantile:float=0.5,linkage:str="ward",dbscan_noise:str="nearest")->dict:
    '''
    T: (N,E) normalized CLIP image embeddings. Returns the tree structure (no VAE stats; see add_node_stats)
    method: "kmeans" (recursive k-means, k fixed or by silhouette), "dbscan" (recursive DBSCAN, falling back to
    2-means when DBSCAN finds fewer than 2 groups), or "agglomerative" (one dendrogram over all images with
    scipy linkage, used as the tree directly, so internal nodes are binary; O(N^2) memory, ~3GB for 28k images)
    in every method, a node with <= c_build images gets one leaf child per image
    on macOS, sklearn and torch can load clashing OpenMP runtimes and segfault; run with OMP_NUM_THREADS=1
    '''
    # imported here so sample.py can use the sampling functions without sklearn
    from sklearn.decomposition import PCA
    T_np=T.float().cpu().numpy()
    N=T_np.shape[0]
    pca=PCA(n_components=min(pca_dim,*T_np.shape),random_state=seed).fit(T_np)
    Y=pca.transform(T_np).astype(np.float32)
    fallbacks=[0]

    # each method describes its items (a node's image set) by: size, the image of a size-1 item, the size-1
    # items of a small node, and how to split a large one
    if method=="agglomerative":
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        Z=scipy_linkage(Y.astype(np.float64),method=linkage)
        root=2*N-2 if N>1 else 0
        def size(c):
            return 1 if c<N else int(Z[c-N,3])
        def leaf_index(c):
            return int(c)
        def leaf_children(c):
            out,stack=[],[c]
            while stack:
                x=stack.pop()
                if x<N:
                    out.append(int(x))
                else:
                    stack+=[int(Z[x-N,1]),int(Z[x-N,0])]
            return out
        def split(c):
            return [int(Z[c-N,0]),int(Z[c-N,1])]
    else:
        root=np.arange(N)
        def size(idx):
            return len(idx)
        def leaf_index(idx):
            return int(idx[0])
        def leaf_children(idx):
            return [np.array([i]) for i in idx]
        def split(idx):
            groups=None
            if method=="dbscan":
                rel=dbscan_groups(Y[idx],dbscan_min_samples,dbscan_quantile,seed,dbscan_noise)
                if rel is None:
                    fallbacks[0]+=1
                    labels=choose_k(Y[idx],"2",k_min,k_max,silhouette_sample,n_init,seed)
                else:
                    groups=[idx[g] for g in rel]
            else:
                labels=choose_k(Y[idx],branching,k_min,k_max,silhouette_sample,n_init,seed)
            if groups is None:
                groups=[idx[labels==l] for l in np.unique(labels)]
            if len(groups)<2:
                # identical embeddings: clustering can't split them, so cut into c_build-sized chunks
                groups=np.array_split(idx,math.ceil(len(idx)/c_build))
            return groups

    # iterative pre-order DFS (dendrograms can be deeper than Python's recursion limit). A node is created when
    # popped and its children are pushed in reverse, so its descendants come next and its images are the
    # contiguous slice perm[start:start+size]
    parent,depth,start,end,children=[],[],[],[],[]
    perm=[]
    stack=[(root,-1,0)]
    while stack:
        item,par,d=stack.pop()
        node=len(parent)
        n=size(item)
        parent.append(par)
        depth.append(d)
        start.append(len(perm))
        end.append(len(perm)+n)
        children.append([])
        if par>=0:
            children[par].append(node)
        if n==1:
            perm.append(leaf_index(item))
            continue
        kids=leaf_children(item) if n<=c_build else split(item)
        for kid in reversed(kids):
            stack.append((kid,node,d+1))
    if fallbacks[0]:
        print(f"dbscan found fewer than 2 groups at {fallbacks[0]} nodes; split those with 2-means")
    return {
        "pca_components":torch.from_numpy(pca.components_.astype(np.float32)),
        "pca_mean":torch.from_numpy(pca.mean_.astype(np.float32)),
        "parent":torch.tensor(parent),
        "depth":torch.tensor(depth),
        "start":torch.tensor(start),
        "end":torch.tensor(end),
        "children":children,
        "perm":torch.tensor(perm),
        "n":N,
        "cluster_method":method
    }


def add_node_stats(tree:dict,X:torch.Tensor,T:torch.Tensor,vae_stats:str="auto",max_bytes:float=4e9,chunk:int=4096)->dict:
    '''
    mean CLIP embedding per node (full and PCA coords), and the mean/variance of the members' bank vectors per
    internal node (only the diagonal node sampler uses those). X may be a large float16 bank: the stats are
    accumulated over row chunks, never converting the whole bank. vae_stats "auto" stores them only if they fit in
    max_bytes (a 256px pixel bank's ~9k internal nodes x 196608 dims would be ~14GB); without them the diagonal
    sampler computes them from the node's images
    '''
    perm,start,end=tree["perm"],tree["start"],tree["end"]
    T_perm=T[perm].float()
    # members are contiguous in perm, so each node is a slice; every image appears once per tree level
    clip_mean=torch.stack([T_perm[s:e].mean(0) for s,e in zip(start.tolist(),end.tolist())])
    internal=torch.tensor([len(c)>0 for c in tree["children"]])
    internal_ids=internal.nonzero().squeeze(-1)
    internal_index=torch.full((len(start),),-1,dtype=torch.long)
    internal_index[internal_ids]=torch.arange(len(internal_ids))
    store=vae_stats=="yes" or (vae_stats=="auto" and 2*len(internal_ids)*X.shape[1]*4<=max_bytes)
    vae_mean=vae_var=None
    if store:
        vae_mean=torch.empty(len(internal_ids),X.shape[1])
        vae_var=torch.empty(len(internal_ids),X.shape[1])
        for k,node in enumerate(internal_ids.tolist()):
            rows=perm[start[node]:end[node]]
            total=torch.zeros(X.shape[1],dtype=torch.float64)
            total_sq=torch.zeros(X.shape[1],dtype=torch.float64)
            for i in range(0,len(rows),chunk):
                x=X[rows[i:i+chunk]].double()
                total+=x.sum(0)
                total_sq+=x.pow(2).sum(0)
            n=len(rows)
            mean=total/n
            vae_mean[k]=mean.float()
            vae_var[k]=((total_sq/n-mean.pow(2)).clamp_min(0)*n/max(n-1,1)).float()
    else:
        print(f"not storing per-node mean/variance ({len(internal_ids)} internal nodes x {X.shape[1]} dims); the diagonal sampler computes them on the fly")
    tree["clip_mean"]=clip_mean
    tree["clip_mean_pca"]=(clip_mean-tree["pca_mean"])@tree["pca_components"].T
    tree["vae_mean"]=vae_mean
    tree["vae_var"]=vae_var
    tree["internal_index"]=internal_index
    return tree


def subset_tree(idx:torch.Tensor)->dict:
    '''
    a one-node "tree" whose only node (0) is the bank rows idx, so sample_node can sample any set of images,
    e.g. the top-K CLIP matches of a prompt (sample.py --method topk)
    '''
    idx=idx.cpu()
    return {"perm":idx,"start":torch.tensor([0]),"end":torch.tensor([len(idx)]),"children":[[]],"parent":torch.tensor([-1]),
            "depth":torch.tensor([0]),"internal_index":torch.tensor([-1]),"vae_mean":None,"vae_var":None,"n":None}


def node_size(tree:dict,node:int)->int:
    return int(tree["end"][node]-tree["start"][node])


def node_members(tree:dict,node:int)->torch.Tensor:
    return tree["perm"][tree["start"][node]:tree["end"][node]]


def descend(tree:dict,text_embed:torch.Tensor,w:float,logit_scale:float,max_depth:int=5,c_search:int=20,
            generator:torch.Generator=None)->int:
    '''
    stochastic descent for one sample: child ~ softmax(w * logit_scale * mean CLIP similarity)
    text_embed: (E,) normalized; similarity is computed in PCA space (projecting the uncentered text
    embedding, so it equals c.(m - mu) up to PCA truncation; the c.mu term is shared by all children)
    '''
    text_pca=tree["pca_components"]@text_embed.float().cpu()
    node=0
    while tree["depth"][node]<max_depth and node_size(tree,node)>c_search and len(tree["children"][node])>0:
        kids=torch.tensor(tree["children"][node])
        scores=tree["clip_mean_pca"][kids]@text_pca
        probs=torch.softmax(w*logit_scale*scores,dim=-1)
        node=int(kids[torch.multinomial(probs,1,generator=generator)])
    return node


def sample_node(tree:dict,X:torch.Tensor,node:int,n:int,node_sampler:str="lowrank",generator:torch.Generator=None,
                cfdm_kwargs:dict=None,locality_kwargs:dict=None,device=None)->torch.Tensor:
    '''
    n samples from one node: diagonal Gaussian (stored mean/var), low-rank Gaussian with the members'
    covariance (same per-dim variance as diagonal, but keeps correlations), CFDM over the members, or
    locality.py's masked analytical model ("locality") / linear Wiener filter ("wiener") with node statistics
    cfdm_kwargs: passed to cfdm.sample, e.g. log_prior (over all bank rows), w, sigma, M, steps, start_t
    ls: Kamb & Ganguli's local score machine (els.py) with a patch-size schedule (locality_kwargs shape, ls_patch_fracs)
    curve: a P-spline curve through the node's images in TSP order, sampled along (curve.py; locality_kwargs["curve"]:
    max_points, n_knots, degree, smooth, grid, mode, tsp, kicks, log_prior, w; the fitted curve is cached per node)
    locality_kwargs: tau, steps, shrinkage, S_global, ridge, log_prior (over all bank rows), w, and cache, a dict
    reused across calls so each node's eigendecomposition is computed once; weights: optional (n_members,) weights
    summing to 1 for the locality/wiener statistics (weighted mean and covariance; the cache must then be per weights)
    device: where to sample (default X's device); X may stay on the CPU (e.g. a float16 256px pixel bank),
    only the node's images are moved
    '''
    device=device or X.device
    members=node_members(tree,node)
    if node_sampler=="wiener" and (locality_kwargs or {}).get("wiener_rank",0)>0 and len(members)>1:
        # low-rank Wiener: statistics streamed from the bank (no d x d matrices, the node never has to fit on the GPU)
        from locality import low_rank_cov,LowRankWiener,wiener_lowrank_sample
        kwargs=locality_kwargs
        cache=kwargs.get("cache",{})
        key=("lowrank",node,kwargs.get("shrinkage",0.0),kwargs.get("ridge",0.0),kwargs["wiener_rank"])
        if key not in cache:
            cache[key]=LowRankWiener(low_rank_cov(X[members.to(X.device)],kwargs["wiener_rank"],device,weights=kwargs.get("weights")),
                                     kwargs.get("global_lowrank"),kwargs.get("shrinkage",0.0),kwargs.get("ridge",0.0))
        return wiener_lowrank_sample(n,cache[key],kwargs.get("steps",10),generator=generator)
    if node_sampler=="curve" and len(members)>1:
        from curve import Curve,curve_sample
        kwargs=dict((locality_kwargs or {}).get("curve") or {})
        cache=(locality_kwargs or {}).get("cache",{})
        max_points=kwargs.get("max_points",2000)
        key=("curve",node,max_points,kwargs.get("n_knots",0),kwargs.get("degree",1),kwargs.get("smooth",0.0),kwargs.get("grid",0),kwargs.get("tsp","auto"))
        if key not in cache:
            # big nodes (e.g. depth 0): the curve goes through a fixed random subset of max_points images
            sub=members
            if len(members)>max_points:
                sub=members[torch.randperm(len(members),generator=torch.Generator().manual_seed(node))[:max_points]]
            built=Curve(X[sub.to(X.device)].to(device).float(),kwargs.get("n_knots",0),kwargs.get("smooth",0.0),
                        kwargs.get("grid",0),kwargs.get("tsp","auto"),kwargs.get("kicks",30),seed=node,degree=kwargs.get("degree",1))
            cache[key]=(built,sub)
        built,sub=cache[key]
        log_prior=kwargs.get("log_prior")
        if log_prior is not None:
            log_prior=torch.log_softmax(log_prior[sub.to(log_prior.device)].to(device).float(),dim=-1)
        return curve_sample(n,built,log_prior,kwargs.get("w",0.0),kwargs.get("mode","random"),kwargs.get("steps",10),generator)
    Xn=X[members.to(X.device)].to(device).float()
    members=members.to(device)
    def randn(*shape):
        return torch.randn(shape,generator=generator,device=device,dtype=Xn.dtype)
    if Xn.shape[0]==1:
        return Xn.expand(n,-1).clone()
    if node_sampler=="diagonal":
        if tree.get("vae_mean") is None:
            var,mean=torch.var_mean(Xn,dim=0)
            return mean+var.sqrt()*randn(n,X.shape[1])
        k=int(tree["internal_index"][node])
        mean=tree["vae_mean"][k].to(device)
        std=tree["vae_var"][k].to(device).sqrt()
        return mean+std*randn(n,X.shape[1])
    if node_sampler=="lowrank":
        mean=Xn.mean(0)
        a=randn(n,Xn.shape[0])
        return mean+a@(Xn-mean)/math.sqrt(Xn.shape[0]-1)
    if node_sampler=="cfdm":
        from cfdm import sample,sample_multiscale
        kwargs=dict(cfdm_kwargs or {})
        log_prior=kwargs.pop("log_prior",None)
        # multiscale: {scales, shape, sigmas, schedule, ...} for sample_multiscale (sigmas replaces sigma)
        multiscale=kwargs.pop("multiscale",None)
        if log_prior is not None:
            log_prior=torch.log_softmax(log_prior[members.to(log_prior.device)].to(device),dim=-1)
        if multiscale:
            kwargs.pop("sigma",None)
            return sample_multiscale(n,Xn,log_prior=log_prior,generator=generator,**kwargs,**multiscale)
        return sample(n,Xn,log_prior,generator=generator,**kwargs)
    if node_sampler in ["locality","wiener"]:
        from locality import node_eig,locality_sample
        kwargs=dict(locality_kwargs or {})
        cache=kwargs.pop("cache",{})
        shrinkage=kwargs.pop("shrinkage",0.0)
        ridge=kwargs.pop("ridge",0.0)
        S_global=kwargs.pop("S_global",None)
        weights=kwargs.pop("weights",None)
        for k in ["spectrum","shape","ls_patch_fracs","wiener_rank","global_lowrank","curve"]:
            kwargs.pop(k,None)
        log_prior=kwargs.pop("log_prior",None)
        if log_prior is not None:
            log_prior=torch.log_softmax(log_prior[members.to(log_prior.device)].to(device),dim=-1)
        if (node,shrinkage,ridge) not in cache:
            cache[(node,shrinkage,ridge)]=node_eig(Xn,S_global,shrinkage,ridge,weights=weights)
        return locality_sample(n,Xn,cache[(node,shrinkage,ridge)],node_sampler,log_prior=log_prior,generator=generator,**kwargs)
    if node_sampler=="ls":
        # Kamb & Ganguli local score machine over the node's images, with a patch-size schedule
        from els import machine_sample,patch_schedule_from_fracs,CELEBA_PATCH_FRACS
        kwargs=dict(locality_kwargs or {})
        shape=kwargs["shape"]
        schedule=patch_schedule_from_fracs(kwargs.get("ls_patch_fracs") or CELEBA_PATCH_FRACS,shape[-1])
        log_prior=kwargs.get("log_prior")
        if log_prior is not None:
            log_prior=torch.log_softmax(log_prior[members.to(log_prior.device)].to(device),dim=-1)
        z0=torch.randn((n,*shape),generator=generator,device=device,dtype=Xn.dtype)
        out=machine_sample("ls",z0,Xn.reshape(-1,*shape),schedule,kwargs.get("steps",10),log_prior,kwargs.get("w",0.0))
        return out.flatten(1)
    raise ValueError(f"unknown node_sampler {node_sampler}")


@torch.no_grad()
def tree_sample(tree:dict,X:torch.Tensor,text_embed:torch.Tensor,n:int,w:float,logit_scale:float,max_depth:int=5,
                c_search:int=20,node_sampler:str="lowrank",generator:torch.Generator=None,cfdm_kwargs:dict=None,
                locality_kwargs:dict=None,device=None):
    '''
    each of the n samples descends independently; samples landing in the same node are drawn together
    returns (n,D) samples and the node id of each; device as in sample_node
    '''
    device=device or X.device
    cpu_generator=None
    if generator is not None:
        cpu_generator=torch.Generator().manual_seed(int(torch.randint(0,2**31,(1,),generator=generator,device=generator.device)))
    nodes=[descend(tree,text_embed,w,logit_scale,max_depth,c_search,cpu_generator) for _ in range(n)]
    out=torch.empty(n,X.shape[1],device=device)
    groups={node:[i for i,m in enumerate(nodes) if m==node] for node in set(nodes)}
    kwargs=locality_kwargs or {}
    if node_sampler in ["locality","wiener"] and kwargs.get("spectrum") is not None and kwargs.get("shrinkage",0)>0 and not kwargs.get("wiener_rank"):
        # Woodbury path: all non-leaf nodes step together and share the global part of the Wiener matrix
        from locality import locality_sample_shrunk,ShrunkNode
        lam,Q=kwargs["spectrum"]
        log_prior=kwargs.get("log_prior")
        cache=kwargs.get("cache")
        inner=[node for node in groups if node_size(tree,node)>1]
        batch=[]
        shrunk=[]
        for node in inner:
            members=node_members(tree,node)
            lp=None if log_prior is None else torch.log_softmax(log_prior[members.to(log_prior.device)].to(device),dim=-1)
            Xn=X[members.to(X.device)].to(device).float()
            batch.append((Xn,len(groups[node]),lp))
            # node statistics are reused across prompts, w and FID batches (big nodes eigendecompose once)
            key=("shrunk",node,kwargs["shrinkage"],kwargs.get("ridge",0.0))
            if cache is None or key not in cache:
                built=ShrunkNode(Xn,lam,Q,kwargs["shrinkage"],kwargs.get("ridge",0.0))
                if cache is None:
                    shrunk.append(built)
                    continue
                cache[key]=built
            shrunk.append(cache[key])
        outs=locality_sample_shrunk(batch,lam,Q,kwargs["shrinkage"],kwargs.get("ridge",0.0),node_sampler,
                                    kwargs.get("tau",0.02),kwargs.get("steps",10),w=kwargs.get("w",0.0),generator=generator,
                                    nodes=shrunk)
        for node,o in zip(inner,outs):
            out[groups[node]]=o
        for node in groups:
            if node_size(tree,node)==1:
                out[groups[node]]=X[node_members(tree,node).to(X.device)].to(device).float().expand(len(groups[node]),-1)
        return out,nodes
    for node,rows in groups.items():
        out[rows]=sample_node(tree,X,node,len(rows),node_sampler,generator,cfdm_kwargs,locality_kwargs,device)
    return out,nodes


def cluster_embeddings(bank:dict,cluster_on:str="image",caption_weight:float=0.5)->torch.Tensor:
    '''
    the (N,E) embeddings a tree is built from: the bank's CLIP image embeddings T, the CLIP text embeddings of its
    captions C, or normalize((1-a) T + a C). All live in CLIP's joint space, so the descent's prompt similarity works
    for each (for captions it is text-text similarity)
    '''
    T=bank["T"].float()
    if cluster_on=="image":
        return T
    if bank.get("C") is None:
        raise ValueError("bank has no caption embeddings C (run caption.py, or embed.py with --label_column)")
    C=bank["C"].float()
    assert C.shape==T.shape,f"caption embeddings {tuple(C.shape)} don't match image embeddings {tuple(T.shape)}"
    if cluster_on=="caption":
        return C
    return torch.nn.functional.normalize((1-caption_weight)*T+caption_weight*C,dim=-1)


def main(args):
    bank=torch.load(args.bank,weights_only=False)
    X=bank["X"]
    T=cluster_embeddings(bank,args.cluster_on,args.caption_weight)
    print(f"bank {args.bank}: X {tuple(X.shape)}, clustering on {args.cluster_on} embeddings {tuple(T.shape)}")
    distinct=len(torch.unique(T,dim=0))
    if distinct<len(T)//2:
        print(f"warning: only {distinct} distinct {args.cluster_on} embeddings for {len(T)} images (template captions?), so clusters below that level are arbitrary")
    start=time.time()
    tree=build_tree(T,args.pca_dim,args.branching,args.k_min,args.k_max,args.silhouette_sample,args.c_build,args.n_init,args.seed,
                    args.cluster_method,args.dbscan_min_samples,args.dbscan_quantile,args.linkage,args.dbscan_noise)
    tree=add_node_stats(tree,X,T,args.vae_stats)
    tree["bank"]=args.bank
    tree["cluster_on"]=args.cluster_on
    tree["args"]=vars(args)
    n_nodes=len(tree["children"])
    n_internal=int((tree["internal_index"]>=0).sum())
    print(f"built tree in {time.time()-start:.1f}s: {n_nodes} nodes, {n_internal} internal, max depth {int(tree['depth'].max())}")
    for d in range(int(tree["depth"].max())+1):
        sizes=(tree["end"]-tree["start"])[tree["depth"]==d].float()
        print(f"\t depth {d}: {len(sizes)} nodes, size min {int(sizes.min())} median {int(sizes.median())} max {int(sizes.max())}")
    os.makedirs(os.path.dirname(args.save_path) or ".",exist_ok=True)
    torch.save(tree,args.save_path)
    print(f"saved {args.save_path}")


if __name__=='__main__':
    from experiment_helpers.gpu_details import print_details
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")
