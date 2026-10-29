import itertools
import math
import numpy as np
import torch
from curve import tsp_path,chained_two_opt,path_length,Curve,curve_sample
from tree import build_tree,add_node_stats,sample_node,node_members
from test_tree import clustered_bank


def arc(m=60,d=20,seed=0,noise=0.0):
    # points along a smooth open 2-D arc, embedded in d dims by a random rotation, returned shuffled
    g=torch.Generator().manual_seed(seed)
    s=torch.linspace(0,1,m)
    pts=torch.stack([torch.cos(3*s),torch.sin(3*s),s],-1)
    Q=torch.linalg.qr(torch.randn(d,d,generator=g))[0][:3]
    X=pts@Q+noise*torch.randn(m,d,generator=g)
    perm=torch.randperm(m,generator=g)
    return X[perm],perm,pts@Q


def test_open_path_recovers_order_along_arc():
    X,perm,_=arc()
    D=torch.cdist(X,X).double().numpy()
    for method in ["2opt","auto"]:
        order=perm[torch.as_tensor(tsp_path(D,method))]
        assert torch.equal(order,torch.arange(len(X))) or torch.equal(order,torch.arange(len(X)).flip(0))


def test_chained_two_opt_is_optimal_on_small_instances():
    rng=np.random.default_rng(0)
    for _ in range(5):
        P=rng.random((8,2))
        D=np.linalg.norm(P[:,None]-P[None],axis=-1)
        tour=chained_two_opt(D,kicks=20)
        length=path_length(np.append(tour,tour[0]),D)
        best=min(path_length(np.array((0,)+p+(0,)),D) for p in itertools.permutations(range(1,8)))
        assert length<=best+1e-9


def test_curve_follows_points():
    X,perm,clean=arc(noise=0.0)
    c=Curve(X,grid=2048)
    # every image is (nearly) on the curve, and the curve stays near the images
    assert torch.cdist(X,c.points).min(-1).values.max()<0.05
    assert torch.cdist(c.points,clean).min(-1).values.max()<0.05
    assert abs(c.length-sum((clean[1:]-clean[:-1]).norm(dim=-1)).item())<1e-3
    # noise well below the spacing between images: the order survives and the smoothed curve stays close
    smooth=Curve(X+0.005*torch.randn(X.shape,generator=torch.Generator().manual_seed(1)),n_knots=16,smooth=1e-2,degree=3)
    assert torch.cdist(smooth.points,clean).min(-1).values.max()<0.05


def test_curve_sample_modes_and_text_prior():
    X,perm,_=arc()
    c=Curve(X)
    g=torch.Generator().manual_seed(0)
    out=curve_sample(500,c,generator=g)
    assert torch.cdist(out,c.points).min(-1).values.max()<1e-3
    # uniform in t: samples spread over the whole curve
    idx=torch.cdist(out,c.points).argmin(-1).float()/len(c.points)
    assert idx.min()<0.05 and idx.max()>0.95
    # a strong text prior on one image pulls samples to its part of the curve
    lp=torch.full((len(X),),-30.0)
    target=int((perm==10).nonzero())
    lp[target]=0.0
    out=curve_sample(200,c,torch.log_softmax(lp,0),w=1.0,generator=g)
    assert (out-X[target]).norm(dim=-1).median()<0.3
    flow=curve_sample(64,c,mode="flow",steps=50,generator=g)
    # the last Euler step averages a few neighboring grid points: within a grid spacing of the curve
    assert torch.cdist(flow,c.points).min(-1).values.max()<c.length/len(c.points)


def test_curve_node_sampler_in_tree():
    X,T,centers,labels=clustered_bank()
    tree=add_node_stats(build_tree(T,pca_dim=8,branching="4",c_build=10,n_init=2),X,T)
    node=tree["children"][0][0]
    cache={}
    kw={"cache":cache,"curve":{"max_points":10,"n_knots":8,"w":0.0}}
    out=sample_node(tree,X,node,6,"curve",torch.Generator().manual_seed(0),locality_kwargs=kw)
    assert out.shape==(6,X.shape[1]) and torch.isfinite(out).all() and len(cache)==1
    built,sub=next(iter(cache.values()))
    assert len(sub)==10 and set(sub.tolist())<=set(node_members(tree,node).tolist())
    sample_node(tree,X,node,3,"curve",locality_kwargs=kw)
    assert len(cache)==1


def test_default_curve_is_the_polyline_through_the_images():
    g=torch.Generator().manual_seed(2)
    X=torch.randn(15,30,generator=g)
    c=Curve(X)
    ordered=X[c.order]
    # every image is on the curve, and so is every crossfade between path neighbors
    assert torch.cdist(X,c.points).min(-1).values.max()<c.length/len(c.points)
    for a in [0.25,0.5]:
        mid=(1-a)*ordered[:-1]+a*ordered[1:]
        assert torch.cdist(mid,c.points).min(-1).values.max()<c.length/len(c.points)
    # and nothing overshoots: every curve point is a convex combination of two neighboring images
    assert c.points.norm(dim=-1).max()<=X.norm(dim=-1).max()+1e-4
