import math
import torch
import pytest
from cfdm import text_log_prior,top_k_subset,k_t,score,velocity,initial_sample,sample
from cfdm import patchify,unpatchify,patch_bank,scale_weights,multiscale_k_t,sample_multiscale


def mixture_log_density(z,X,t,log_prior):
    # log sum_i p_i N(z; t x_i, (1-t)^2 I), up to a z-independent constant
    d2=(z[:,None,:]-t*X[None]).pow(2).sum(-1)
    return torch.logsumexp(log_prior-d2/(2*(1-t)**2),dim=-1)


@pytest.fixture
def setup():
    g=torch.Generator().manual_seed(0)
    N,D,E=20,6,5
    X=torch.randn(N,D,generator=g,dtype=torch.float64)
    T=torch.nn.functional.normalize(torch.randn(N,E,generator=g,dtype=torch.float64),dim=-1)
    c=torch.nn.functional.normalize(torch.randn(E,generator=g,dtype=torch.float64),dim=-1)
    return X,T,c,g


@pytest.mark.parametrize("t",[0.1,0.5,0.9])
def test_score_matches_autograd(setup,t):
    X,T,c,g=setup
    log_prior=text_log_prior(c,T,10.0)
    z=torch.randn(4,X.shape[1],generator=g,dtype=torch.float64,requires_grad=True)
    grad,=torch.autograd.grad(mixture_log_density(z,X,t,log_prior).sum(),z)
    assert torch.allclose(score(z.detach(),X,t,log_prior,w=1.0),grad,atol=1e-8)


def test_cosine_equals_squared_distance(setup):
    # ||c - tau||^2 = 2 - 2cos, so the prior matches the -||c - T||^2 formulation
    X,T,c,g=setup
    s=7.0
    via_dist=torch.log_softmax(-s/2*(c[None]-T).pow(2).sum(-1),dim=-1)
    assert torch.allclose(text_log_prior(c,T,s),via_dist,atol=1e-10)


def test_w0_is_unconditional(setup):
    X,T,c,g=setup
    z=torch.randn(3,X.shape[1],generator=g,dtype=torch.float64)
    log_prior=text_log_prior(c,T,50.0)
    assert torch.allclose(k_t(z,X,0.4,log_prior,w=0.0),k_t(z,X,0.4,None),atol=1e-12)


def test_large_w_collapses_to_best_match(setup):
    X,T,c,g=setup
    log_prior=text_log_prior(c,T,1.0)
    best=log_prior.argmax()
    z=torch.randn(3,X.shape[1],generator=g,dtype=torch.float64)
    out=k_t(z,X,0.3,log_prior,w=1e6)
    assert torch.allclose(out,0.3*X[best].expand_as(out),atol=1e-8)


def test_top_k_full_is_identity(setup):
    X,T,c,g=setup
    log_prior=text_log_prior(c,T,10.0)
    X_sub,lp_sub,idx=top_k_subset(X,log_prior,X.shape[0])
    assert torch.equal(X_sub,X) and torch.allclose(lp_sub,log_prior)
    X_sub,lp_sub,idx=top_k_subset(X,log_prior,5)
    assert X_sub.shape[0]==5 and torch.isclose(lp_sub.exp().sum(),torch.tensor(1.0,dtype=lp_sub.dtype))
    assert idx[0]==log_prior.argmax()


def test_velocity_matches_paper_form(setup):
    X,T,c,g=setup
    log_prior=text_log_prior(c,T,10.0)
    z=torch.randn(3,X.shape[1],generator=g,dtype=torch.float64)
    t=0.6
    paper=(z+(1-t)*score(z,X,t,log_prior))/t
    assert torch.allclose(velocity(z,X,t,log_prior),paper,atol=1e-10)


def test_initial_sample_follows_prior(setup):
    X,T,c,g=setup
    log_prior=torch.full((X.shape[0],),-math.inf,dtype=torch.float64)
    log_prior[3]=0.0
    z=initial_sample(1000,X,0.999999,log_prior,w=1.0,generator=g)
    assert torch.allclose(z.mean(0),0.999999*X[3],atol=1e-3)


@pytest.mark.parametrize("start_t",[0.0,0.5])
def test_unsmoothed_sampling_memorizes(setup,start_t):
    # with sigma=0 the model is the empirical distribution, so samples land on training points
    X,T,c,g=setup
    out=sample(16,X,None,steps=200,start_t=start_t,generator=g)
    d=torch.cdist(out,X).min(-1).values
    assert (d<1e-3).all()


def test_strong_text_guidance_picks_prompt_match(setup):
    X,T,c,g=setup
    log_prior=text_log_prior(c,T,100.0)
    out=sample(16,X,log_prior,w=10.0,steps=100,start_t=0.2,generator=g)
    nearest=torch.cdist(out,X).argmin(-1)
    assert (nearest==log_prior.argmax()).float().mean()>0.9


def test_smoothing_leaves_training_set(setup):
    X,T,c,g=setup
    out=sample(16,X,None,sigma=1.0,M=4,steps=100,start_t=0.2,generator=g)
    assert torch.isfinite(out).all()
    assert (torch.cdist(out,X).min(-1).values>1e-3).any()


SHAPE=(2,8,8)


@pytest.fixture
def image_setup():
    g=torch.Generator().manual_seed(1)
    N=12
    X=torch.randn(N,2*8*8,generator=g,dtype=torch.float64)
    T=torch.nn.functional.normalize(torch.randn(N,5,generator=g,dtype=torch.float64),dim=-1)
    c=torch.nn.functional.normalize(torch.randn(5,generator=g,dtype=torch.float64),dim=-1)
    return X,T,c,g


@pytest.mark.parametrize("s",[1,2,4])
def test_patchify_roundtrip(image_setup,s):
    X,T,c,g=image_setup
    assert torch.equal(unpatchify(patchify(X,SHAPE,s),SHAPE,s),X)


def test_patchify_grid_order(image_setup):
    X,T,c,g=image_setup
    img=X.reshape(-1,*SHAPE)
    p=patchify(X,SHAPE,2)
    # patch 1 is the top-right quarter, patch 2 the bottom-left
    assert torch.equal(p[:,1],img[:,:,:4,4:].reshape(X.shape[0],-1))
    assert torch.equal(p[:,2],img[:,:,4:,:4].reshape(X.shape[0],-1))


def test_patch_prior_inherits_parent(image_setup):
    X,T,c,g=image_setup
    log_prior=text_log_prior(c,T,10.0)
    patches,patch_prior=patch_bank(X,SHAPE,4,log_prior)
    assert patches.shape==(X.shape[0]*16,8)
    # each image keeps its total prior mass, split evenly over its 16 patches
    assert torch.allclose(patch_prior.exp().reshape(-1,16).sum(-1),log_prior.exp())


def test_scale_weights_schedule():
    assert scale_weights(0.2,3,"coarse_to_fine",0.2)==[1.0,0.0,0.0]
    assert scale_weights(1.0,3,"coarse_to_fine",0.2)==[0.0,0.0,1.0]
    assert scale_weights(0.6,3,"coarse_to_fine",0.2)==pytest.approx([0.0,1.0,0.0])
    assert scale_weights(0.2,3,"fine_to_coarse",0.2)==[0.0,0.0,1.0]
    assert scale_weights(0.5,3,"constant",constant=[2,1,1])==[0.5,0.25,0.25]
    for t in [0.25,0.4,0.77,0.95]:
        assert sum(scale_weights(t,3,"coarse_to_fine",0.2))==pytest.approx(1.0)


def test_scale_one_matches_whole_image(image_setup):
    X,T,c,g=image_setup
    log_prior=text_log_prior(c,T,10.0)
    z=torch.randn(3,X.shape[1],generator=g,dtype=torch.float64)
    patches,patch_prior=patch_bank(X,SHAPE,1,log_prior)
    bank={"s":1,"shape":SHAPE,"patches":patches,"log_prior":patch_prior,"x_sq":patches.pow(2).sum(-1)}
    assert torch.allclose(multiscale_k_t(z,[bank],0.5,[1.0],w=2.0),k_t(z,X,0.5,log_prior,w=2.0),atol=1e-12)


def test_finest_scale_makes_collages(image_setup):
    # with sigma=0 and only 16ths, every output patch is some training patch (from any position),
    # and a sample generally mixes patches from several images
    X,T,c,g=image_setup
    out=sample_multiscale(8,X,SHAPE,scales=[4],steps=200,start_t=0.2,generator=g)
    patches,_=patch_bank(X,SHAPE,4)
    out_p=patchify(out,SHAPE,4).reshape(-1,8)
    d,nearest=torch.cdist(out_p,patches).min(-1)
    assert (d<1e-3).all()
    parents=(nearest//16).reshape(8,16)
    assert max(len(set(row.tolist())) for row in parents)>1


def test_multiscale_one_scale_equals_sample(image_setup):
    X,T,c,g=image_setup
    log_prior=text_log_prior(c,T,10.0)
    a=sample(4,X,log_prior,w=1.0,steps=30,start_t=0.2,generator=torch.Generator().manual_seed(5))
    b=sample_multiscale(4,X,SHAPE,scales=[1],log_prior=log_prior,w=1.0,steps=30,start_t=0.2,generator=torch.Generator().manual_seed(5))
    assert torch.allclose(a,b,atol=1e-10)


def test_same_position_matches_any_with_one_scale(image_setup):
    X,T,c,g=image_setup
    log_prior=text_log_prior(c,T,10.0)
    a=sample_multiscale(4,X,SHAPE,scales=[1],log_prior=log_prior,steps=30,start_t=0.2,positions="any",generator=torch.Generator().manual_seed(5))
    b=sample_multiscale(4,X,SHAPE,scales=[1],log_prior=log_prior,steps=30,start_t=0.2,positions="same",generator=torch.Generator().manual_seed(5))
    assert torch.allclose(a,b,atol=1e-10)


def test_same_position_keeps_positions(image_setup):
    X,T,c,g=image_setup
    out=sample_multiscale(8,X,SHAPE,scales=[4],steps=200,start_t=0.0,positions="same",generator=g)
    cand,_=patch_bank(X,SHAPE,4)
    d,nearest=torch.cdist(patchify(out,SHAPE,4).reshape(-1,8),cand).min(-1)
    assert (d<1e-3).all()
    assert torch.equal((nearest%16).reshape(8,16),torch.arange(16).expand(8,-1))


def test_same_position_k_t_matches_per_region_k_t(image_setup):
    X,T,c,g=image_setup
    log_prior=text_log_prior(c,T,10.0)
    z=torch.randn(3,X.shape[1],generator=g,dtype=torch.float64)
    patches=patchify(X,SHAPE,2)
    bank={"s":2,"shape":SHAPE,"positions":"same","patches":patches,"log_prior":log_prior,"x_sq":patches.pow(2).sum(-1)}
    out=patchify(multiscale_k_t(z,[bank],0.4,[1.0],w=1.5),SHAPE,2)
    z_p=patchify(z,SHAPE,2)
    for j in range(4):
        assert torch.allclose(out[:,j],k_t(z_p[:,j],patches[:,j],0.4,log_prior,w=1.5),atol=1e-12)
