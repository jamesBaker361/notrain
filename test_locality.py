import torch
import pytest
from locality import node_eig,wiener_matrix,binarize,masked_denoise,wiener_denoise,locality_sample
from cfdm import k_t
from tree import build_tree,add_node_stats,sample_node,tree_sample


@pytest.fixture
def data():
    g=torch.Generator().manual_seed(0)
    Xn=torch.randn(30,8,generator=g,dtype=torch.float64)@torch.randn(8,8,generator=g,dtype=torch.float64)
    z=torch.randn(4,8,generator=g,dtype=torch.float64)
    return Xn,z,g


def test_binarize_keeps_row_max():
    W=torch.tensor([[1.0,-0.5,0.01],[0.2,-3.0,0.0],[0.0,0.0,0.0]],dtype=torch.float64)
    M=binarize(W,0.4)
    assert torch.equal(M,torch.tensor([[1.,1.,0.],[0.,1.,0.],[1.,1.,1.]],dtype=torch.float64))
    assert torch.equal(binarize(W,0.0),torch.ones(3,3,dtype=torch.float64))


@pytest.mark.parametrize("t",[0.2,0.6,0.9])
def test_full_mask_is_ideal_denoiser(data,t):
    # every pixel sees the whole image -> the exact whole-image posterior mean
    Xn,z,g=data
    out=masked_denoise(z,Xn,t,torch.ones(8,8,dtype=torch.float64),chunk=7)
    assert torch.allclose(out,k_t(z,Xn,t)/t,atol=1e-10)


def test_identity_mask_is_per_pixel(data):
    Xn,z,g=data
    t=0.5
    out=masked_denoise(z,Xn,t,torch.eye(8,dtype=torch.float64),chunk=4)
    for q in range(8):
        wts=torch.softmax(-(z[:,q:q+1]-t*Xn[None,:,q]).pow(2)/(2*(1-t)**2),dim=-1)
        assert torch.allclose(out[:,q],wts@Xn[:,q],atol=1e-10)


def test_text_prior_in_masked_denoise(data):
    Xn,z,g=data
    log_prior=torch.log_softmax(torch.randn(30,generator=g,dtype=torch.float64),-1)
    out=masked_denoise(z,Xn,0.4,torch.ones(8,8,dtype=torch.float64),log_prior,w=2.0,chunk=8)
    assert torch.allclose(out,k_t(z,Xn,0.4,log_prior,w=2.0)/0.4,atol=1e-10)


def test_wiener_is_gaussian_posterior_mean(data):
    Xn,z,g=data
    mean,evals,evecs,floor=node_eig(Xn)
    S=torch.cov(Xn.T)
    t=0.3
    s2=((1-t)/t)**2
    direct=mean+(S@torch.linalg.solve(S+s2*torch.eye(8,dtype=torch.float64),(z/t-mean).T)).T
    assert torch.allclose(wiener_denoise(z,t,mean,evals,evecs,floor),direct,atol=1e-8)


def test_wiener_matrix_limits(data):
    Xn,z,g=data
    mean,evals,evecs,floor=node_eig(Xn)
    assert torch.allclose(wiener_matrix(evals,evecs,1e-12),torch.eye(8,dtype=torch.float64),atol=1e-6)
    assert wiener_matrix(evals,evecs,1e12).abs().max()<1e-9


def test_shrinkage(data):
    Xn,z,g=data
    S_global=torch.eye(8,dtype=torch.float64)*2
    _,evals,evecs,_=node_eig(Xn[:3],S_global,1.0)
    assert torch.allclose((evecs*evals)@evecs.T,S_global,atol=1e-10)
    # a 3-image node has rank 2 without shrinkage
    _,evals,_,_=node_eig(Xn[:3])
    assert (evals>1e-8).sum()==2


def test_ridge_makes_masks_local_at_low_noise(data):
    Xn,z,g=data
    _,evals,evecs,floor=node_eig(Xn[:3],ridge=0.01)
    assert torch.equal(binarize(wiener_matrix(evals,evecs,1e-10,floor),0.5),torch.eye(8,dtype=torch.float64))
    _,evals,evecs,floor=node_eig(Xn[:3])
    assert binarize(wiener_matrix(evals,evecs,1e-10,floor),0.5).sum()>8


@pytest.mark.parametrize("ridge",[0.0,0.05])
def test_lowrank_path_matches_full_eig(data,ridge):
    # n < d: the SVD path must give the same Wiener matrix and filter as the d x d eigendecomposition
    Xn,z,g=data
    Xs=Xn[:5]
    full=node_eig(Xs,ridge=ridge,lowrank=False)
    low=node_eig(Xs,ridge=ridge,lowrank=True)
    assert low[2].shape==(8,5) and full[2].shape==(8,8)
    for s2 in [0.01,1.0,100.0]:
        assert torch.allclose(wiener_matrix(*low[1:3],s2,low[3]),wiener_matrix(*full[1:3],s2,full[3]),atol=1e-9)
    assert torch.allclose(wiener_denoise(z,0.3,*low),wiener_denoise(z,0.3,*full),atol=1e-9)


def test_tau0_sampling_memorizes(data):
    Xn,z,g=data
    out=locality_sample(6,Xn,node_eig(Xn),"locality",tau=0.0,steps=200,generator=g)
    assert torch.cdist(out,Xn).min(-1).values.max()<1e-3


def test_locality_and_wiener_sampling_finite(data):
    Xn,z,g=data
    for kind in ["locality","wiener"]:
        out=locality_sample(5,Xn,node_eig(Xn),kind,tau=0.02,steps=10,generator=g)
        assert out.shape==(5,8) and torch.isfinite(out).all()


def test_tree_node_samplers_with_cache():
    g=torch.Generator().manual_seed(0)
    T=torch.nn.functional.normalize(torch.randn(60,6,generator=g),dim=-1)
    X=torch.randn(60,8,generator=g)
    tree=add_node_stats(build_tree(T,pca_dim=4,branching="3",c_build=10,n_init=1),X,T)
    cache={}
    for kind in ["locality","wiener"]:
        out,nodes=tree_sample(tree,X,T[0],6,w=1.0,logit_scale=10.0,max_depth=2,c_search=20,node_sampler=kind,
                              generator=g,locality_kwargs={"tau":0.02,"steps":5,"cache":cache})
        assert out.shape==(6,8) and torch.isfinite(out).all()
    assert len(cache)>=1
    S_global=torch.cov(X.T)
    node=tree["children"][0][0]
    out=sample_node(tree,X,node,3,"locality",g,locality_kwargs={"steps":5,"shrinkage":0.5,"S_global":S_global,"cache":cache})
    assert torch.isfinite(out).all() and (node,0.5,0.0) in cache


from locality import global_spectrum,ShrunkNode,locality_sample_shrunk


@pytest.fixture
def shrink_data():
    g=torch.Generator().manual_seed(3)
    d=10
    Xall=torch.randn(200,d,generator=g,dtype=torch.float64)@torch.randn(d,d,generator=g,dtype=torch.float64)
    S_global=torch.cov(Xall.T)
    return Xall,S_global,global_spectrum(S_global),g


@pytest.mark.parametrize("n",[4,30])
@pytest.mark.parametrize("shrinkage,ridge",[(0.3,0.0),(0.7,0.05),(1.0,0.0)])
def test_woodbury_matches_eig(shrink_data,n,shrinkage,ridge):
    Xall,S_global,(lam,Q),g=shrink_data
    Xn=Xall[:n]
    exact=node_eig(Xn,S_global,shrinkage,ridge)
    node=ShrunkNode(Xn,lam,Q,shrinkage,ridge)
    r=torch.randn(3,10,generator=g,dtype=torch.float64)
    for s2 in [1e-3,0.1,1.0,100.0]:
        W=wiener_matrix(exact[1],exact[2],s2,exact[3])
        assert torch.allclose(node.wiener_matrix(s2),W,atol=1e-8)
        assert torch.allclose(node.wiener_apply(r,s2),r@W.T,atol=1e-8)


@pytest.mark.parametrize("kind",["locality","wiener"])
def test_woodbury_sampler_matches_eig_sampler(shrink_data,kind):
    # same noise, same steps: the lockstep Woodbury sampler reproduces the per-node eigendecomposition sampler
    Xall,S_global,(lam,Q),g=shrink_data
    Xn=Xall[:12]
    a=locality_sample(5,Xn,node_eig(Xn,S_global,0.5,0.01),kind,tau=0.05,steps=10,generator=torch.Generator().manual_seed(9))
    b=locality_sample_shrunk([(Xn,5,None)],lam,Q,0.5,0.01,kind,tau=0.05,steps=10,generator=torch.Generator().manual_seed(9))[0]
    assert torch.allclose(a,b,atol=1e-6)


def test_woodbury_lockstep_several_nodes(shrink_data):
    Xall,S_global,(lam,Q),g=shrink_data
    outs=locality_sample_shrunk([(Xall[:6],2,None),(Xall[6:40],3,None)],lam,Q,0.5,0.0,"locality",steps=5,generator=g)
    assert [o.shape for o in outs]==[(2,10),(3,10)] and all(torch.isfinite(o).all() for o in outs)


def test_tree_sample_woodbury_path():
    g=torch.Generator().manual_seed(0)
    T=torch.nn.functional.normalize(torch.randn(60,6,generator=g),dim=-1)
    X=torch.randn(60,8,generator=g)
    tree=add_node_stats(build_tree(T,pca_dim=4,branching="3",c_build=10,n_init=1),X,T)
    spectrum=global_spectrum(torch.cov(X.T))
    for kind in ["locality","wiener"]:
        out,nodes=tree_sample(tree,X,T[0],6,w=1.0,logit_scale=10.0,max_depth=2,c_search=20,node_sampler=kind,generator=g,
                              locality_kwargs={"tau":0.02,"steps":5,"shrinkage":0.5,"ridge":0.0,"spectrum":spectrum})
        assert out.shape==(6,8) and torch.isfinite(out).all()


def test_woodbury_lockstep_no_nodes(shrink_data):
    # every sample landed on a leaf: nothing to run
    Xall,S_global,(lam,Q),g=shrink_data
    assert locality_sample_shrunk([],lam,Q,0.5,0.0,"locality")==[]
