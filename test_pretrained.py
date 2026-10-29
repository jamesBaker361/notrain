import math
import torch
import pytest
from pretrained import DiffusionNet,TrainedNet,diffusion_sample

BASE="google/ddpm-cifar10-32"


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return DiffusionNet(BASE,"scratch").eval()


def test_t_min_is_noisiest_training_point(net):
    a=net.alphas_cumprod[-1].sqrt().item()
    assert net.t_min==pytest.approx(a/(a+math.sqrt(1-a*a)))
    assert 0<net.t_min<0.01


@pytest.mark.parametrize("k",[10,250,999])
def test_x0_matches_ddpm_at_integer_timestep(net,k):
    abar=net.alphas_cumprod[k]
    a,b=abar.sqrt().item(),(1-abar).sqrt().item()
    x_k=torch.randn(2,3,32,32)
    t=a/(a+b)
    z=x_k*math.sqrt(t*t+(1-t)**2)
    with torch.no_grad():
        assert net.timestep(abar[None])[0].item()==pytest.approx(k,abs=1e-3)
        direct=((x_k-b*net.unet(x_k,torch.tensor([k,k])).sample)/a).clamp(-1,1)
        # x0 divides by sqrt(abar): float32 rounding is amplified ~1/a at the noisy end
        assert torch.allclose(net.x0(z,t),direct,atol=max(1e-4,1e-5/a))


def test_zero_init_adapters_change_nothing(net):
    netx=DiffusionNet(BASE,"scratch",cross_attention=True,token_dim=16).eval()
    netx.unet.load_state_dict(net.unet.state_dict())
    z=torch.randn(2,3,32,32)
    text=torch.randn(2,5,16)
    mask=torch.ones(2,5)
    with torch.no_grad():
        assert len(netx.adapters)>0
        assert torch.equal(netx.x0(z,0.4,text,mask,text,mask,w=2.0),net.x0(z,0.4))
    # once the adapters are non-zero, text changes the prediction and w interpolates
    for p in netx.adapters.parameters():
        torch.nn.init.normal_(p,std=0.02)
    with torch.no_grad():
        null=torch.zeros(2,5,16)
        c=netx.x0(z,0.4,text,mask,null,mask,w=1.0)
        u=netx.x0(z,0.4,text,mask,null,mask,w=0.0)
        assert not torch.allclose(c,u)


def test_save_load_roundtrip(net,tmp_path):
    netx=DiffusionNet(BASE,"scratch",cross_attention=True,token_dim=16)
    for p in netx.adapters.parameters():
        torch.nn.init.normal_(p,std=0.02)
    netx.save(str(tmp_path/"m"),{"step":1})
    loaded,meta=DiffusionNet.load(str(tmp_path/"m"))
    assert meta["diffusion_net"]["cross_attention"] and meta["step"]==1
    for a,b in zip(netx.state_dict().values(),loaded.state_dict().values()):
        assert torch.equal(a,b)


def test_trained_net_unconditional_uses_empty_caption(tmp_path):
    netx=DiffusionNet(BASE,"scratch",cross_attention=True,token_dim=16)
    for p in netx.adapters.parameters():
        torch.nn.init.normal_(p,std=0.02)
    netx.save(str(tmp_path/"m"))
    calls=[]
    def clip_fn(texts):
        calls.append(texts)
        g=torch.Generator().manual_seed(len(texts[0]))
        return torch.randn(1,5,16,generator=g),torch.ones(1,5)
    tn=TrainedNet(str(tmp_path/"m"),"cpu",clip_fn)
    assert tn.conditional and tn.t_min==pytest.approx(netx.t_min)
    z0=torch.randn(2,3,32,32)
    out=tn.sample(z0,steps=2,t_min=tn.t_min)
    assert out.shape==z0.shape and [""] in calls
    # matches explicitly sampling with the empty caption at w=0
    null_text,null_mask=clip_fn([""])
    ref=diffusion_sample(tn.model,z0,2,tn.t_min,null_text.expand(2,-1,-1),null_mask.expand(2,-1),null_text.expand(2,-1,-1),null_mask.expand(2,-1),0.0)
    assert torch.allclose(out,ref,atol=1e-6)


def test_celebahq_architecture_at_64(tmp_path):
    # fully convolutional: the 256px celebahq UNet runs on 64x64 faces (bottom level 2x2)
    net=DiffusionNet("google/ddpm-celebahq-256","scratch",cross_attention=True,token_dim=16,sample_size=64).eval()
    assert net.unet.config.sample_size==64 and len(net.adapters)>0
    with torch.no_grad():
        out=net.x0(torch.randn(1,3,64,64),0.5,torch.randn(1,5,16),torch.ones(1,5),torch.zeros(1,5,16),torch.ones(1,5),w=2.0)
    assert out.shape==(1,3,64,64) and torch.isfinite(out).all()
    net.save(str(tmp_path/"m"))
    loaded,meta=DiffusionNet.load(str(tmp_path/"m"))
    assert meta["diffusion_net"]["sample_size"]==64 and loaded.unet.config.sample_size==64


def test_sample_size_must_divide():
    with pytest.raises(AssertionError):
        DiffusionNet("google/ddpm-celebahq-256","scratch",sample_size=48)


def test_scratch_channels_override(tmp_path):
    # the LDM UNet on 4x32x32 SD-VAE latents instead of its 3x64x64 VQ latents
    net=DiffusionNet("CompVis/ldm-celebahq-256","scratch",cross_attention=True,token_dim=16,sample_size=32,channels=4).eval()
    assert net.unet.config.in_channels==4 and net.unet.config.out_channels==4
    with torch.no_grad():
        out=net.x0(torch.randn(1,4,32,32),0.5,torch.randn(1,5,16),torch.ones(1,5),torch.zeros(1,5,16),torch.ones(1,5),w=2.0)
    assert out.shape==(1,4,32,32) and torch.isfinite(out).all()
    net.save(str(tmp_path/"m"))
    loaded,meta=DiffusionNet.load(str(tmp_path/"m"))
    assert meta["diffusion_net"]["channels"]==4 and loaded.unet.config.in_channels==4
    with pytest.raises(AssertionError):
        DiffusionNet("CompVis/ldm-celebahq-256","pretrained",channels=4)
