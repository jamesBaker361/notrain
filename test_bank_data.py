import os
import torch
from PIL import Image
from datasets import Dataset,ClassLabel
from bank import load_images,metadata_strings,compose_caption


def test_metadata_strings():
    ds=Dataset.from_dict({"artist":[0,1,2],"tag":["Blood Moon","",None]})
    ds=ds.cast_column("artist",ClassLabel(names=["Unknown Artist","claude-monet","tench, Tinca tinca"]))
    assert metadata_strings(ds,"artist")==["","claude monet","tench"]
    assert metadata_strings(ds,"tag")==["Blood Moon","",""]


def test_compose_caption_tidies_empty_fields():
    t="{blip}, {style} {genre} by {artist}"
    assert compose_caption(t,blip="a woman in a garden",style="Impressionism",genre="portrait",artist="claude monet")=="a woman in a garden, Impressionism portrait by claude monet"
    assert compose_caption(t,blip="a woman in a garden",style="Impressionism",genre="",artist="")=="a woman in a garden, Impressionism"
    assert compose_caption(t,blip="a river",style="",genre="",artist="")=="a river"
    assert compose_caption("{blip}, {tag} {champion} splash art",blip="a man with a sword",tag="",champion="Aatrox")=="a man with a sword, Aatrox splash art"
    assert compose_caption("{blip}",blip="a dog")=="a dog"


def test_load_images_shards_and_shuffle_are_reproducible(tmp_path):
    # caption.py must see the same rows in the same order as embed.py did
    for shard in range(3):
        os.makedirs(tmp_path/f"part{shard}")
        for i in range(5):
            Image.new("RGB",(4,4),(shard*80,i*50,0)).save(tmp_path/f"part{shard}"/f"{i}.png")
    files=[str(tmp_path/"part0"/"*.png"),str(tmp_path/"part1"/"*.png")]
    pixel=lambda ds:[im.getpixel((0,0)) for im in ds["image"]]
    a=load_images("imagefolder","train",limit=7,data_files=files,shuffle=True,seed=1)
    b=load_images("imagefolder","train",limit=7,data_files=files,shuffle=True,seed=1)
    assert len(a)==7 and pixel(a)==pixel(b)
    assert all(p[0] in (0,80) for p in pixel(a))
    full=load_images("imagefolder","train",data_files=files)
    assert len(full)==10 and sorted(pixel(full))==sorted(pixel(load_images("imagefolder","train",data_files=files,shuffle=True)))


def test_unsharp_mask():
    from bank import unsharp_mask
    g=torch.Generator().manual_seed(0)
    flat=torch.full((2,3,16,16),0.2)
    assert torch.allclose(unsharp_mask(flat,1.0,1.0),flat,atol=1e-6)
    # a blurry step edge gets steeper (more contrast across it), stays in range, and the shape is kept
    edge=torch.zeros(1,1,16,16)
    edge[...,8:]=1.0
    edge=edge*1.6-0.8
    soft=torch.nn.functional.avg_pool2d(torch.nn.functional.pad(edge,(1,1,1,1),mode="replicate"),3,1)
    sharp=unsharp_mask(soft,1.5,1.0)
    assert sharp.shape==soft.shape and sharp.abs().max()<=1
    assert (sharp[...,8]-sharp[...,7]).mean()>(soft[...,8]-soft[...,7]).mean()
    # threshold leaves small noise alone
    noisy=0.2+0.01*torch.randn(1,3,16,16,generator=g)
    assert torch.allclose(unsharp_mask(noisy,1.0,1.0,threshold=0.05),noisy)
