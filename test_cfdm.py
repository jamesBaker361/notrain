import math
import torch
import pytest
from cfdm import text_log_prior,top_k_subset,k_t,score,velocity,initial_sample,sample


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
