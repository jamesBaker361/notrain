import torch
import torch.nn.functional as F
import pytest
from pyramid import (pyramid_sizes,downsample_flat,gaussian_window,patch_scores,build_patch_bank,patch_denoise,
                     image_denoise,blend,build_pyramid_banks,sample_pyramid,resize)


@pytest.fixture
def images():
    g=torch.Generator().manual_seed(0)
    # smooth random images, so downsampling keeps most of their content
    x=torch.randn(4,3,8,8,generator=g)
    return resize(x,(16,16)),g


def test_pyramid_sizes():
    assert pyramid_sizes(64,64,16)==[(64,64),(32,32),(16,16)]
    assert pyramid_sizes(32,32,8)==[(32,32),(16,16),(8,8)]
    assert pyramid_sizes(16,16,16)==[(16,16)]


def test_patch_count_and_prior(images):
    x,g=images
    prior=torch.log_softmax(torch.randn(4,generator=g),-1)
    bank=build_patch_bank(x,5,prior,w=2.0)
    L=(16-5+1)**2
    assert bank["patches"].shape==(4*L,3*25)
    assert torch.allclose(bank["bias"],2.0*prior.repeat_interleave(L))


def test_patch_scores_match_unfold(images):
    x,g=images
    heat=torch.rand(4,1,16,16,generator=g)
    assert torch.allclose(patch_scores(heat,5),F.unfold(heat,5).mean(1))


def test_heatmap_filter_keeps_top_and_random(images):
    x,g=images
    heat=torch.rand(4,1,16,16,generator=g)
    full=build_patch_bank(x,5,heat=heat,heatmap_lambda=1.0)
    score=full["bias"]
    M=score.numel()
    top=build_patch_bank(x,5,heat=heat,heatmap_lambda=1.0,keep=0.25)
    n_keep=-(-M//4)
    assert top["patches"].shape[0]==n_keep
    # exactly the highest-scoring patches
    assert torch.allclose(torch.sort(top["bias"]).values,torch.sort(torch.topk(score,n_keep).values).values)
    mixed=build_patch_bank(x,5,heat=heat,heatmap_lambda=1.0,keep=0.25,random_frac=0.5,generator=torch.Generator().manual_seed(1))
    assert mixed["patches"].shape[0]==n_keep+round(0.5*(M-n_keep))
    assert mixed["bias"].min()<top["bias"].min()


def test_heatmap_lambda_standardizes(images):
    x,g=images
    heat=torch.rand(4,1,16,16,generator=g)*100+7
    bank=build_patch_bank(x,5,heat=heat,heatmap_lambda=1.0)
    assert abs(bank["bias"].mean())<1e-5 and abs(bank["bias"].std()-1)<1e-4


def test_patch_denoise_chunks_match_softmax(images):
    x,g=images
    prior=torch.log_softmax(torch.randn(4,generator=g),-1)
    bank=build_patch_bank(x,5,prior,w=1.0)
    q=torch.randn(37,75,generator=g)
    t=0.6
    logits=-torch.cdist(q,t*bank["patches"]).pow(2)/(2*(1-t)**2)+bank["bias"]
    expected=torch.softmax(logits,-1)@bank["patches"]
    assert torch.allclose(patch_denoise(q,bank,t,max_elements=bank["patches"].shape[0]*5),expected,atol=1e-5)


def test_image_denoise_reproduces_clean_image(images):
    # near t=1 every patch of a clean bank image snaps to itself, and the windowed fold puts them back exactly
    x,g=images
    bank=build_patch_bank(x,5)
    window=gaussian_window(3,5,0.2)
    out=image_denoise(x[:2],bank,0.999,5,window)
    assert torch.allclose(out,x[:2],atol=1e-4)


def test_blend_with_own_downsample_is_identity(images):
    x,g=images
    assert torch.allclose(blend(x,resize(x,(8,8))),x,atol=1e-5)


def test_blend_takes_low_frequencies_from_coarser(images):
    x,g=images
    other=torch.randn(4,3,8,8,generator=g)
    out=blend(x,other)
    # replacing the low band moves the result toward up(other); the detail band of x is kept
    detail=x-resize(resize(x,(8,8)),(16,16))
    assert torch.allclose(out-resize(other,(16,16)),detail,atol=1e-5)


def test_sample_pyramid_shapes_and_memorization(images):
    # one image, no smoothing, strong patch prior: the coarse CFDM returns its downsample and the patch levels
    # rebuild it from its own patches
    x,g=images
    shape=(3,16,16)
    X=x[:1].flatten(1)
    coarse=downsample_flat(X,shape,(8,8),"cpu")
    banks=build_pyramid_banks(x[:1],8,5)
    assert [b["patches"].shape[0] for b in banks]==[(16-5+1)**2]
    out,levels=sample_pyramid(3,coarse,shape,banks,8,5,sigma=0.0,cfdm_steps=20,start_t=0.5,steps=20,
                              generator=g,return_levels=True)
    assert out.shape==(3,3*16*16)
    assert [tuple(l.shape) for l in levels]==[(3,3,8,8),(3,3,16,16)]
    assert torch.isfinite(out).all()
    err=(out-X).norm(dim=-1)/X.norm()
    assert (err<0.1).all(),err


def test_sample_pyramid_mixes_images(images):
    # two images: the coarse CFDM can pick either, and the result stays near the patch manifold of the bank
    x,g=images
    shape=(3,16,16)
    X=x.flatten(1)
    coarse=downsample_flat(X,shape,(8,8),"cpu")
    banks=build_pyramid_banks(x,8,5)
    out=sample_pyramid(8,coarse,shape,banks,8,5,sigma=0.0,cfdm_steps=20,start_t=0.5,steps=20,generator=g)
    q=F.unfold(out.reshape(-1,*shape),5).transpose(1,2).reshape(-1,75)
    nearest=torch.cdist(q,banks[0]["patches"]).min(-1).values
    noise=torch.cdist(torch.randn(200,75,generator=g),banks[0]["patches"]).min(-1).values
    assert nearest.mean()<0.25*noise.mean()
