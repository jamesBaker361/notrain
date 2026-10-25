'''
hierarchical CLIP tree over a bank built by embed.py

build: PCA on the CLIP image embeddings T, then recursive k-means in PCA space. A node with more than
--c_build images is split by k-means (k fixed, or chosen by silhouette score); a node with --c_build or
fewer gets one leaf child per image, so every leaf holds exactly one image. Members of every node are
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


def build_tree(T:torch.Tensor,pca_dim:int=64,branching:str="10",k_min:int=2,k_max:int=10,silhouette_sample:int=2000,
               c_build:int=10,n_init:int=3,seed:int=123)->dict:
    '''
    T: (N,E) normalized CLIP image embeddings. Returns the tree structure (no VAE stats; see add_node_stats)
    on macOS, sklearn and torch can load clashing OpenMP runtimes and segfault; run with OMP_NUM_THREADS=1
    '''
    # imported here so sample.py can use the sampling functions without sklearn
    from sklearn.decomposition import PCA
    T_np=T.float().cpu().numpy()
    pca=PCA(n_components=min(pca_dim,*T_np.shape),random_state=seed).fit(T_np)
    Y=pca.transform(T_np).astype(np.float32)

    parent,depth,start,end,children=[],[],[],[],[]
    perm=[]

    def new_node(par:int,d:int)->int:
        parent.append(par)
        depth.append(d)
        start.append(len(perm))
        end.append(-1)
        children.append([])
        return len(parent)-1

    def build(idx:np.ndarray,par:int,d:int)->int:
        node=new_node(par,d)
        if len(idx)==1:
            perm.append(int(idx[0]))
        elif len(idx)<=c_build:
            for i in idx:
                children[node].append(build(np.array([i]),node,d+1))
        else:
            labels=choose_k(Y[idx],branching,k_min,k_max,silhouette_sample,n_init,seed)
            groups=[idx[labels==l] for l in np.unique(labels)]
            if len(groups)<2:
                # identical embeddings: k-means can't split them, so cut into c_build-sized chunks
                groups=np.array_split(idx,math.ceil(len(idx)/c_build))
            for g in groups:
                children[node].append(build(g,node,d+1))
        end[node]=len(perm)
        return node

    build(np.arange(T_np.shape[0]),-1,0)
    return {
        "pca_components":torch.from_numpy(pca.components_.astype(np.float32)),
        "pca_mean":torch.from_numpy(pca.mean_.astype(np.float32)),
        "parent":torch.tensor(parent),
        "depth":torch.tensor(depth),
        "start":torch.tensor(start),
        "end":torch.tensor(end),
        "children":children,
        "perm":torch.tensor(perm),
        "n":T_np.shape[0]
    }


def add_node_stats(tree:dict,X:torch.Tensor,T:torch.Tensor)->dict:
    '''
    mean CLIP embedding per node (full and PCA coords), and VAE latent mean/variance per internal node
    '''
    perm,start,end=tree["perm"],tree["start"],tree["end"]
    X_perm=X[perm].float()
    T_perm=T[perm].float()
    # members are contiguous in perm, so each node is a slice; every image appears once per tree level
    clip_mean=torch.stack([T_perm[s:e].mean(0) for s,e in zip(start.tolist(),end.tolist())])
    internal=torch.tensor([len(c)>0 for c in tree["children"]])
    internal_ids=internal.nonzero().squeeze(-1)
    internal_index=torch.full((len(start),),-1,dtype=torch.long)
    internal_index[internal_ids]=torch.arange(len(internal_ids))
    vae_mean=torch.empty(len(internal_ids),X.shape[1])
    vae_var=torch.empty(len(internal_ids),X.shape[1])
    for k,node in enumerate(internal_ids.tolist()):
        vae_var[k],vae_mean[k]=torch.var_mean(X_perm[start[node]:end[node]],dim=0)
    tree["clip_mean"]=clip_mean
    tree["clip_mean_pca"]=(clip_mean-tree["pca_mean"])@tree["pca_components"].T
    tree["vae_mean"]=vae_mean
    tree["vae_var"]=vae_var
    tree["internal_index"]=internal_index
    return tree


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
                cfdm_kwargs:dict=None,locality_kwargs:dict=None)->torch.Tensor:
    '''
    n samples from one node: diagonal Gaussian (stored mean/var), low-rank Gaussian with the members'
    covariance (same per-dim variance as diagonal, but keeps correlations), CFDM over the members, or
    locality.py's masked analytical model ("locality") / linear Wiener filter ("wiener") with node statistics
    cfdm_kwargs: passed to cfdm.sample, e.g. log_prior (over all bank rows), w, sigma, M, steps, start_t
    locality_kwargs: tau, steps, shrinkage, S_global, ridge, log_prior (over all bank rows), w, and cache, a dict
    reused across calls so each node's eigendecomposition is computed once
    '''
    members=node_members(tree,node).to(X.device)
    Xn=X[members]
    def randn(*shape):
        return torch.randn(shape,generator=generator,device=X.device,dtype=X.dtype)
    if Xn.shape[0]==1:
        return Xn.expand(n,-1).clone()
    if node_sampler=="diagonal":
        k=int(tree["internal_index"][node])
        mean=tree["vae_mean"][k].to(X.device)
        std=tree["vae_var"][k].to(X.device).sqrt()
        return mean+std*randn(n,X.shape[1])
    if node_sampler=="lowrank":
        mean=Xn.mean(0)
        a=randn(n,Xn.shape[0])
        return mean+a@(Xn-mean)/math.sqrt(Xn.shape[0]-1)
    if node_sampler=="cfdm":
        from cfdm import sample
        kwargs=dict(cfdm_kwargs or {})
        log_prior=kwargs.pop("log_prior",None)
        if log_prior is not None:
            log_prior=torch.log_softmax(log_prior[members],dim=-1)
        return sample(n,Xn,log_prior,generator=generator,**kwargs)
    if node_sampler in ["locality","wiener"]:
        from locality import node_eig,locality_sample
        kwargs=dict(locality_kwargs or {})
        cache=kwargs.pop("cache",{})
        shrinkage=kwargs.pop("shrinkage",0.0)
        ridge=kwargs.pop("ridge",0.0)
        S_global=kwargs.pop("S_global",None)
        kwargs.pop("spectrum",None)
        log_prior=kwargs.pop("log_prior",None)
        if log_prior is not None:
            log_prior=torch.log_softmax(log_prior[members],dim=-1)
        if (node,shrinkage,ridge) not in cache:
            cache[(node,shrinkage,ridge)]=node_eig(Xn,S_global,shrinkage,ridge)
        return locality_sample(n,Xn,cache[(node,shrinkage,ridge)],node_sampler,log_prior=log_prior,generator=generator,**kwargs)
    raise ValueError(f"unknown node_sampler {node_sampler}")


@torch.no_grad()
def tree_sample(tree:dict,X:torch.Tensor,text_embed:torch.Tensor,n:int,w:float,logit_scale:float,max_depth:int=5,
                c_search:int=20,node_sampler:str="lowrank",generator:torch.Generator=None,cfdm_kwargs:dict=None,
                locality_kwargs:dict=None):
    '''
    each of the n samples descends independently; samples landing in the same node are drawn together
    returns (n,D) samples and the node id of each
    '''
    cpu_generator=None
    if generator is not None:
        cpu_generator=torch.Generator().manual_seed(int(torch.randint(0,2**31,(1,),generator=generator,device=generator.device)))
    nodes=[descend(tree,text_embed,w,logit_scale,max_depth,c_search,cpu_generator) for _ in range(n)]
    out=torch.empty(n,X.shape[1],device=X.device,dtype=X.dtype)
    groups={node:[i for i,m in enumerate(nodes) if m==node] for node in set(nodes)}
    kwargs=locality_kwargs or {}
    if node_sampler in ["locality","wiener"] and kwargs.get("spectrum") is not None and kwargs.get("shrinkage",0)>0:
        # Woodbury path: all non-leaf nodes step together and share the global part of the Wiener matrix
        from locality import locality_sample_shrunk
        lam,Q=kwargs["spectrum"]
        log_prior=kwargs.get("log_prior")
        inner=[node for node in groups if node_size(tree,node)>1]
        batch=[]
        for node in inner:
            members=node_members(tree,node).to(X.device)
            lp=None if log_prior is None else torch.log_softmax(log_prior[members],dim=-1)
            batch.append((X[members],len(groups[node]),lp))
        outs=locality_sample_shrunk(batch,lam,Q,kwargs["shrinkage"],kwargs.get("ridge",0.0),node_sampler,
                                    kwargs.get("tau",0.02),kwargs.get("steps",10),w=kwargs.get("w",0.0),generator=generator)
        for node,o in zip(inner,outs):
            out[groups[node]]=o
        for node in groups:
            if node_size(tree,node)==1:
                out[groups[node]]=X[node_members(tree,node).to(X.device)].expand(len(groups[node]),-1)
        return out,nodes
    for node,rows in groups.items():
        out[rows]=sample_node(tree,X,node,len(rows),node_sampler,generator,cfdm_kwargs,locality_kwargs)
    return out,nodes


def main(args):
    bank=torch.load(args.bank,weights_only=False)
    X,T=bank["X"],bank["T"]
    print(f"bank {args.bank}: X {tuple(X.shape)} T {tuple(T.shape)}")
    start=time.time()
    tree=build_tree(T,args.pca_dim,args.branching,args.k_min,args.k_max,args.silhouette_sample,args.c_build,args.n_init,args.seed)
    tree=add_node_stats(tree,X,T)
    tree["bank"]=args.bank
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
