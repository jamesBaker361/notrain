import torch
import torch.nn.functional as F
import pytest
from pyramid import (pyramid_sizes,downsample_flat,gaussian_window,token_overlap,token_to_patches,build_patch_bank,patch_denoise,
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


def test_token_overlap():
    # 16px side, 4 tokens of 4px, 5px windows: every window covers 5 pixels in total
    O=token_overlap(16,4,5)
    assert O.shape==(12,4) and torch.allclose(O.sum(1),torch.full((12,),5.0))
    assert O[0].tolist()==[4,1,0,0] and O[3].tolist()==[1,4,0,0] and O[4].tolist()==[0,4,1,0]
    # token edges between pixels (20px, 16 tokens): fractional overlaps, still summing to the window
    assert torch.allclose(token_overlap(20,16,5).sum(1),torch.full((16,),5.0))


def test_token_to_patches_is_mean_over_covered_pixels(images):
    x,g=images
    token=torch.rand(4,4,4,generator=g)
    # each token painted over its 4x4 pixel cell: a patch's score is the mean of its pixels
    painted=token.repeat_interleave(4,1).repeat_interleave(4,2)[:,None]
    assert torch.allclose(token_to_patches(token,(16,16),5),F.unfold(painted,5).mean(1),atol=1e-6)
    # one token per pixel: the plain mean over each patch's footprint
    assert torch.allclose(token_to_patches(painted[:,0],(16,16),5),F.unfold(painted,5).mean(1),atol=1e-6)


def test_heatmap_filter_keeps_patches_overlapping_kept_tokens(images):
    x,g=images
    heat=torch.rand(4,4,4,generator=g)
    full=build_patch_bank(x,5,heat=heat,heatmap_lambda=1.0)
    M=full["patches"].shape[0]
    top=build_patch_bank(x,5,heat=heat,heatmap_lambda=1.0,keep=0.25)
    # the top quarter of the 64 tokens, and every 5x5 patch touching one of their 4x4 cells
    kept=torch.zeros(64)
    kept[torch.topk(heat.flatten(),16).indices]=1
    painted=kept.reshape(4,1,4,4).repeat_interleave(4,2).repeat_interleave(4,3)
    covered=F.unfold(painted,5).amax(1).flatten()>0
    assert top["patches"].shape[0]==covered.sum()
    assert torch.equal(torch.sort(top["bias"]).values,torch.sort(full["bias"][covered]).values)
    mixed=build_patch_bank(x,5,heat=heat,heatmap_lambda=1.0,keep=0.25,random_frac=0.5,generator=torch.Generator().manual_seed(1))
    assert mixed["patches"].shape[0]==covered.sum()+round(0.5*(M-covered.sum().item()))


def test_heatmap_filter_min_overlap(images):
    x,g=images
    heat=torch.rand(4,4,4,generator=g)
    kept=torch.zeros(64)
    kept[torch.topk(heat.flatten(),16).indices]=1
    painted=kept.reshape(4,1,4,4).repeat_interleave(4,2).repeat_interleave(4,3)
    # the fraction of each 5x5 patch's pixels inside kept tokens
    frac=F.unfold(painted,5).mean(1).flatten()
    for min_overlap in [0.25,0.5]:
        bank=build_patch_bank(x,5,heat=heat,keep=0.25,min_overlap=min_overlap)
        assert bank["patches"].shape[0]==(frac>=min_overlap-1e-6).sum()>0
    # stricter than any overlapping pixel
    assert (frac>=0.5).sum()<(frac>0).sum()
    # never empty: an unreachable threshold falls back to the best-covered patches
    one=torch.zeros(64)
    one[0]=1
    bank=build_patch_bank(x,5,heat=one.reshape(4,4,4),keep=1/64,min_overlap=1.0)
    painted=one.reshape(4,1,4,4).repeat_interleave(4,2).repeat_interleave(4,3)
    frac=F.unfold(painted,5).mean(1).flatten()
    assert bank["patches"].shape[0]==(frac>=frac.max()-1e-6).sum()>0


def test_heatmap_lambda_standardizes(images):
    x,g=images
    heat=torch.rand(4,4,4,generator=g)*100+7
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
