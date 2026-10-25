import torch
import pytest
from metrics import precision_recall,prompt_accuracy,knn_radii
from els import patch_schedule_from_fracs,CELEBA_PATCH_FRACS
from cfdm import sample
from tree import build_tree,add_node_stats,sample_node
from textunet import TextUNet,net_sample


def test_precision_recall_limits():
    g=torch.Generator().manual_seed(0)
    real=torch.randn(200,8,generator=g)
    # same distribution, same points: everything is covered both ways
    assert precision_recall(real,real.clone(),k=3)==(1.0,1.0)
    # samples far from the data: no precision, no recall
    p,r=precision_recall(real,real+100,k=3)
    assert p==0.0 and r==0.0
    # samples collapsed onto a few real points: precise but low coverage
    p,r=precision_recall(real,real[:5].repeat(40,1)+1e-4*torch.randn(200,8,generator=g),k=3)
    assert p==1.0 and r<0.2


def test_knn_radii_excludes_self():
    x=torch.tensor([[0.],[1.],[3.]])
    assert torch.allclose(knn_radii(x,1),torch.tensor([1.,1.,2.]))


def test_prompt_accuracy():
    text=torch.eye(3)
    images=torch.tensor([[1.,0,0],[0,1.,0],[1.,0,0]])
    assert prompt_accuracy(images,text,torch.tensor([0,1,2]))==pytest.approx(2/3)


def test_lpips_identical_is_zero():
    from metrics import LpipsDiversity
    m=LpipsDiversity("cpu","squeeze")
    x=torch.rand(1,3,32,32)*2-1
    assert m.mean_pairwise(x.repeat(3,1,1,1))<1e-6
    assert m.mean_pairwise(torch.rand(3,3,32,32)*2-1)>0.01
    assert m.mean_pairwise(x)is None


def test_patch_schedule():
    sched=patch_schedule_from_fracs(CELEBA_PATCH_FRACS,64)
    assert sched(1e-3)==65 and sched(0.999)==3
    sizes={sched(t) for t in torch.linspace(1e-3,0.999,50).tolist()}
    assert all(s%2==1 for s in sizes)
    # smaller images scale down
    assert patch_schedule_from_fracs(CELEBA_PATCH_FRACS,32)(0.999)==3


def test_cfdm_z0_matches_internal_noise():
    g=torch.Generator().manual_seed(0)
    X=torch.randn(20,6,generator=g,dtype=torch.float64)
    a=sample(4,X,None,steps=20,start_t=0.0,generator=torch.Generator().manual_seed(5))
    z0=torch.randn((4,6),generator=torch.Generator().manual_seed(5),dtype=torch.float64)
    b=sample(4,X,None,steps=20,start_t=0.0,z0=z0)
    assert torch.allclose(a,b)


def test_ls_node_sampler():
    g=torch.Generator().manual_seed(0)
    T=torch.nn.functional.normalize(torch.randn(40,6,generator=g),dim=-1)
    X=torch.randn(40,3*8*8,generator=g)
    tree=add_node_stats(build_tree(T,pca_dim=4,branching="3",c_build=10,n_init=1),X,T)
    node=tree["children"][0][0]
    out=sample_node(tree,X,node,3,"ls",g,locality_kwargs={"shape":(3,8,8),"steps":5})
    assert out.shape==(3,192) and torch.isfinite(out).all()


def test_net_sample_unconditional():
    model=TextUNet(in_channels=3,sample_size=8,block_out_channels=(16,32),layers_per_block=1,text_dim=1,norm_num_groups=8)
    out=net_sample(model,torch.randn(2,3,8,8),steps=3)
    assert out.shape==(2,3,8,8) and torch.isfinite(out).all()
