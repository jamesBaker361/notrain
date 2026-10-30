import torch
from locality import node_eig,low_rank_cov,weighted_center
from tree import subset_tree,sample_node,node_members


def bank(n=40,d=12,seed=0):
    g=torch.Generator().manual_seed(seed)
    return torch.randn(n,d,generator=g)@torch.randn(d,d,generator=g),g


def weighted_cov(X,w):
    mean=w@X
    Xc=X-mean
    return mean,len(X)/(len(X)-1)*(Xc.T*w)@Xc


def test_weighted_center_uniform_is_unweighted():
    X,_=bank()
    m0,c0=weighted_center(X)
    m1,c1=weighted_center(X,torch.full((len(X),),1/len(X)))
    assert torch.allclose(m0,m1,atol=1e-5) and torch.allclose(c0.T@c0,c1.T@c1,atol=1e-3)


def test_weighted_node_eig_and_low_rank_cov():
    X,g=bank()
    w=torch.softmax(torch.randn(len(X),generator=g)*2,0)
    mean,S=weighted_cov(X,w)
    for lowrank in [True,False]:
        m,evals,evecs,floor=node_eig(X,lowrank=lowrank,weights=w)
        assert torch.allclose(m,mean,atol=1e-4)
        assert torch.allclose(evecs@torch.diag(evals)@evecs.T,S,rtol=1e-3,atol=1e-3)
    m,evals,V,resid=low_rank_cov(X,12,"cpu",chunk=7,weights=w,generator=torch.Generator().manual_seed(0))
    assert torch.allclose(m,mean,atol=1e-4) and torch.allclose(V@torch.diag(evals)@V.T,S,rtol=1e-3,atol=1e-2)


def test_subset_tree_samplers():
    X,g=bank(60,16)
    idx=torch.tensor([3,7,11,20,21,30,41,50])
    tree=subset_tree(idx)
    assert torch.equal(node_members(tree,0),idx)
    lp=torch.log_softmax(torch.randn(len(X),generator=g),0)
    for sampler in ["cfdm","wiener","locality","curve","lowrank","diagonal"]:
        out=sample_node(tree,X,0,5,sampler,torch.Generator().manual_seed(0),
                        cfdm_kwargs={"log_prior":lp,"w":1.0,"sigma":0.0,"steps":20},
                        locality_kwargs={"steps":10,"cache":{},"weights":torch.full((8,),1/8),"curve":{"log_prior":lp,"w":1.0}})
        assert out.shape==(5,16) and torch.isfinite(out).all(),sampler
    # unsmoothed cfdm only returns members of the set
    out=sample_node(tree,X,0,5,"cfdm",torch.Generator().manual_seed(0),cfdm_kwargs={"sigma":0.0,"steps":50})
    assert torch.cdist(out,X[idx]).min(-1).values.max()<1e-3


def test_weights_pull_wiener_toward_heavy_image():
    X,g=bank(60,16)
    idx=torch.arange(10)
    w=torch.full((10,),1e-4)
    w[4]=1.0
    w=w/w.sum()
    out=sample_node(subset_tree(idx),X,0,64,"wiener",torch.Generator().manual_seed(0),
                    locality_kwargs={"steps":10,"cache":{},"weights":w})
    uniform=sample_node(subset_tree(idx),X,0,64,"wiener",torch.Generator().manual_seed(0),
                        locality_kwargs={"steps":10,"cache":{},"weights":torch.full((10,),0.1)})
    assert (out.mean(0)-X[4]).norm()<0.5*(uniform.mean(0)-X[4]).norm()
