import torch
import torch.nn.functional as F
import pytest
from els import ideal_denoise,ls_denoise,els_denoise,r2,machine_sample,text_log_prior_from


@pytest.fixture
def data():
    g=torch.Generator().manual_seed(0)
    X=torch.randn(7,2,6,6,generator=g,dtype=torch.float64)
    z=torch.randn(3,2,6,6,generator=g,dtype=torch.float64)
    log_prior=torch.log_softmax(torch.randn(7,generator=g,dtype=torch.float64),dim=-1)
    return X,z,log_prior,g


def brute_force(z,X,t,P,log_prior,w,equivariant):
    B,C,H,W=z.shape
    N,R=X.shape[0],H*W
    q=F.unfold(z,P,padding=P//2).transpose(1,2)            # (B,R,D)
    cand=F.unfold(X,P,padding=P//2).transpose(1,2)         # (N,R,D)
    centers=X.permute(0,2,3,1).reshape(N,R,C)
    out=torch.empty(B,R,C,dtype=z.dtype)
    for b in range(B):
        for r in range(R):
            if equivariant:
                c,cen,lp=cand.reshape(N*R,-1),centers.reshape(N*R,C),log_prior.repeat_interleave(R)
            else:
                c,cen,lp=cand[:,r],centers[:,r],log_prior
            logits=-(q[b,r]-t*c).pow(2).sum(-1)/(2*(1-t)**2)+w*lp
            out[b,r]=torch.softmax(logits,-1)@cen
    return out.transpose(1,2).reshape(B,C,H,W)


@pytest.mark.parametrize("t",[0.2,0.7])
@pytest.mark.parametrize("P",[1,3,5])
def test_ls_matches_brute_force(data,t,P):
    X,z,log_prior,g=data
    assert torch.allclose(ls_denoise(z,X,t,P,log_prior,w=1.5,chunk=3),brute_force(z,X,t,P,log_prior,1.5,False),atol=1e-10)


@pytest.mark.parametrize("P",[1,3])
def test_els_matches_brute_force(data,P):
    X,z,log_prior,g=data
    out=els_denoise(z,X,0.5,P,log_prior,w=1.5,query_chunk=7,cand_chunk=50)
    assert torch.allclose(out,brute_force(z,X,0.5,P,log_prior,1.5,True),atol=1e-10)


def test_ls_with_full_window_is_ideal(data):
    # a window covering the whole image from every pixel sees every image in full
    X,z,log_prior,g=data
    assert torch.allclose(ls_denoise(z,X,0.6,13,log_prior,w=2.0),ideal_denoise(z,X,0.6,log_prior,w=2.0),atol=1e-10)


def test_large_w_returns_prior_argmax(data):
    X,z,log_prior,g=data
    best=X[log_prior.argmax()]
    for out in [ideal_denoise(z,X,0.3,log_prior,w=1e6),ls_denoise(z,X,0.3,3,log_prior,w=1e6)]:
        assert torch.allclose(out,best.expand_as(out),atol=1e-8)


def test_r2():
    target=torch.randn(4,10,dtype=torch.float64)
    assert torch.allclose(r2(target,target),torch.ones(4,dtype=torch.float64))
    assert torch.allclose(r2(target.mean(-1,keepdim=True).expand_as(target),target),torch.zeros(4,dtype=torch.float64))


def test_ideal_sampling_memorizes(data):
    X,z,log_prior,g=data
    out=machine_sample("ideal",z,X,None,steps=100)
    assert torch.cdist(out.flatten(1),X.flatten(1)).min(-1).values.max()<1e-3


def test_ls_sampling_runs_and_is_finite(data):
    X,z,log_prior,g=data
    out=machine_sample("ls",z,X,lambda t:3 if t<0.5 else 1,steps=20,log_prior=log_prior)
    assert out.shape==z.shape and torch.isfinite(out).all()


def test_text_log_prior_from_normalizes():
    keys=F.normalize(torch.randn(5,4,dtype=torch.float64),dim=-1)
    c=F.normalize(torch.randn(2,4,dtype=torch.float64),dim=-1)
    lp=text_log_prior_from(c,keys,10.0)
    assert lp.shape==(2,5) and torch.allclose(lp.exp().sum(-1),torch.ones(2,dtype=torch.float64))
