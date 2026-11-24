'''
helpers for building and decoding the "bank": training points X (flattened pixels, SD-VAE latents, the
VQ latents of CompVis/ldm-celebahq-256, or FLUX.2 latents) and their CLIP image embeddings T
'''
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPModel, AutoProcessor
from diffusers import AutoencoderKL,VQModel


def _features(out)->torch.Tensor:
    # transformers>=5 returns BaseModelOutputWithPooling whose pooler_output is the projected embedding
    if isinstance(out,torch.Tensor):
        return out
    return out.pooler_output


def load_clip(clip_model:str,device,torch_dtype=torch.float32):
    model=CLIPModel.from_pretrained(clip_model,torch_dtype=torch_dtype).to(device).eval()
    processor=AutoProcessor.from_pretrained(clip_model)
    return model,processor


@torch.no_grad()
def encode_images_clip(model:CLIPModel,processor,images:list,device)->torch.Tensor:
    inputs=processor(images=images,return_tensors="pt").to(device)
    feats=_features(model.get_image_features(pixel_values=inputs["pixel_values"].to(model.dtype)))
    return F.normalize(feats.float(),dim=-1)


@torch.no_grad()
def encode_texts_clip(model:CLIPModel,processor,texts:list,device)->torch.Tensor:
    inputs=processor(text=texts,return_tensors="pt",padding=True,truncation=True).to(device)
    feats=_features(model.get_text_features(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"]))
    return F.normalize(feats.float(),dim=-1)


@torch.no_grad()
def encode_texts_clip_tokens(model:CLIPModel,processor,texts:list,device):
    '''
    per-token CLIP text states for cross-attention: (B,77,hidden) and the (B,77) attention mask
    '''
    inputs=processor(text=texts,return_tensors="pt",padding="max_length",truncation=True,max_length=77).to(device)
    states=model.text_model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"]).last_hidden_state
    return states.float(),inputs["attention_mask"]


def load_vae(vae_name:str,device,torch_dtype=torch.float32,space:str="vae"):
    '''
    space "vae": an AutoencoderKL repo (e.g. stabilityai/sd-vae-ft-mse); space "vq": the VQ autoencoder in the
    vqvae subfolder of a latent-diffusion repo (e.g. CompVis/ldm-celebahq-256); space "flux2": the
    AutoencoderKLFlux2 in the vae subfolder of black-forest-labs/FLUX.2-dev (gated: accept its license on the Hub
    and log in; needs diffusers>=0.36)
    '''
    if space=="vq":
        return VQModel.from_pretrained(vae_name,subfolder="vqvae",torch_dtype=torch_dtype).to(device).eval()
    if space=="flux2":
        from diffusers import AutoencoderKLFlux2
        return AutoencoderKLFlux2.from_pretrained(vae_name,subfolder="vae",torch_dtype=torch_dtype).to(device).eval()
    return AutoencoderKL.from_pretrained(vae_name,torch_dtype=torch_dtype).to(device).eval()


def load_bank_vae(bank:dict,device,torch_dtype=torch.float32):
    # the autoencoder a bank was built with, or None for a pixel bank
    if bank["space"]=="pixel":
        return None
    return load_vae(bank["vae"],device,torch_dtype,bank["space"])


def _is_flux2(vae)->bool:
    # by name, so older diffusers without AutoencoderKLFlux2 still import this module
    return type(vae).__name__=="AutoencoderKLFlux2"


def _flux2_stats(vae,like:torch.Tensor):
    '''
    FLUX.2 normalizes its 2x2-patchified latents (4*32 channels) with the VAE's BatchNorm running statistics
    instead of a scaling factor (Flux2Pipeline); returns that mean and std, (1,128,1,1)
    '''
    mean=vae.bn.running_mean.view(1,-1,1,1).to(like)
    std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps).to(like)
    return mean,std


@torch.no_grad()
def encode_vae(vae,pixels:torch.Tensor)->torch.Tensor:
    # pixels in [-1,1]
    if _is_flux2(vae):
        # normalized exactly as Flux2Pipeline does, then un-patchified back to (32,H/8,W/8): the same values the
        # FLUX.2 transformer sees, on the finer 8x grid, which leaves the pyramid sampler more levels
        r=vae.config.patch_size[0]
        assert pixels.shape[-1]%(8*r)==0 and pixels.shape[-2]%(8*r)==0,f"FLUX.2 latents need sides divisible by {8*r}"
        latents=vae.encode(pixels.to(vae.dtype)).latent_dist.mean.float()
        mean,std=_flux2_stats(vae,latents)
        return F.pixel_shuffle((F.pixel_unshuffle(latents,r)-mean)/std,r)
    if isinstance(vae,VQModel):
        # pre-quantization latents, the space latent diffusion samples in (LDMPipeline decodes them unscaled)
        return vae.encode(pixels.to(vae.dtype)).latents.float()
    # posterior mean so the bank is deterministic
    latents=vae.encode(pixels.to(vae.dtype)).latent_dist.mean
    return latents.float()*vae.config.scaling_factor


@torch.no_grad()
def decode_vae(vae,latents:torch.Tensor)->torch.Tensor:
    if _is_flux2(vae):
        r=vae.config.patch_size[0]
        mean,std=_flux2_stats(vae,latents.float())
        latents=F.pixel_shuffle(F.pixel_unshuffle(latents.float(),r)*std+mean,r)
        return vae.decode(latents.to(vae.dtype)).sample.float()
    if isinstance(vae,VQModel):
        return vae.decode(latents.to(vae.dtype)).sample.float()
    return vae.decode((latents/vae.config.scaling_factor).to(vae.dtype)).sample.float()


def to_pil(pixels:torch.Tensor)->list:
    # (B,C,H,W) in [-1,1]
    arr=((pixels.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
    return [Image.fromarray(a) for a in arr]


def unsharp_mask(pixels:torch.Tensor,amount:float=1.0,radius:float=1.0,threshold:float=0.0)->torch.Tensor:
    '''
    pixels: (B,C,H,W) in [-1,1] -> sharpened, same range: x + amount * (x - gaussian_blur(x, sigma=radius pixels)),
    only where |x - blur| exceeds threshold (in [0,1] intensity units, as in PIL's UnsharpMask)
    '''
    x=pixels.float()
    half=max(1,int(math.ceil(3*radius)))
    k=torch.exp(-0.5*(torch.arange(-half,half+1,device=x.device,dtype=x.dtype)/radius)**2)
    k=k/k.sum()
    C=x.shape[1]
    # separable blur, reflect padding so edges aren't darkened
    blur=F.pad(x,(half,half,0,0),mode="reflect")
    blur=F.conv2d(blur,k.view(1,1,1,-1).expand(C,1,1,-1),groups=C)
    blur=F.pad(blur,(0,0,half,half),mode="reflect")
    blur=F.conv2d(blur,k.view(1,1,-1,1).expand(C,1,-1,1),groups=C)
    detail=x-blur
    if threshold>0:
        detail=detail*(detail.abs()>2*threshold)
    return (x+amount*detail).clamp(-1,1)


def row_chunk(X:torch.Tensor,max_bytes:float=2**29)->int:
    # bank rows per chunk so a float32 copy of the chunk stays under max_bytes (4096 rows of 256px pixels are 3GB)
    return max(1,int(max_bytes//(X[0].numel()*4)))


def bank_to_pixels(bank:dict,flat:torch.Tensor,vae:AutoencoderKL=None,batch_size:int=8)->torch.Tensor:
    '''
    flat: (B,D) points in bank space -> (B,C,H,W) pixels in [-1,1]
    '''
    x=flat.reshape(-1,*bank["shape"])
    if bank["space"]=="pixel":
        # pixel banks may be stored in float16
        return x.float()
    return torch.cat([decode_vae(vae,x[i:i+batch_size].to(vae.device)) for i in range(0,len(x),batch_size)]).cpu()


def load_images(dataset:str,split:str,limit:int=-1,data_files:list=None,shuffle:bool=False,seed:int=123):
    '''
    the HF image dataset rows a bank is built from. caption.py calls this with the bank's stored embed.py args,
    so it gets the same rows in the same order. data_files (globs) loads only some shards, e.g. part of WikiArt;
    shuffle (before limit) matters for class-sorted datasets like ImageNet, where the first rows are a few classes
    '''
    from datasets import load_dataset
    # a subset of shards doesn't match the split sizes recorded in the dataset card, so skip that check
    ds=load_dataset(dataset,split=split,data_files=data_files,verification_mode="no_checks") if data_files else load_dataset(dataset,split=split)
    if shuffle:
        ds=ds.shuffle(seed=seed)
    if limit>0:
        ds=ds.select(range(min(limit,len(ds))))
    return ds


def pixel_transform(resolution:int):
    # PIL image -> (3,R,R) in [-1,1], as embed.py stores every bank
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])


def held_out_pixels(bank:dict,limit:int=-1)->torch.Tensor:
    '''
    the real images behind the bank's X_test, in order, as (n,3,R,R) pixels in [-1,1] (embed.py's transform):
    a latent bank's X_test only decodes to its autoencoder's reconstructions, these are the images themselves
    '''
    a=bank["args"]
    if a.get("test_split"):
        ds=load_images(a["dataset"],a["test_split"],a.get("test_limit",-1),a.get("data_files"),a.get("shuffle",False),a.get("seed",123))
    else:
        # embed.py --holdout: the last rows of the training split
        ds=load_images(a["dataset"],a["split"],a["limit"],a.get("data_files"),a.get("shuffle",False),a.get("seed",123))
        ds=ds.select(range(bank["X"].shape[0],len(ds)))
    assert len(ds)==bank["X_test"].shape[0],f"{len(ds)} held-out dataset rows for {bank['X_test'].shape[0]} X_test rows"
    n=len(ds) if limit<=0 else min(limit,len(ds))
    transform=pixel_transform(a["resolution"])
    return torch.stack([transform(img.convert("RGB")) for img in ds.select(range(n))[a["image_column"]]])


def bank_dataset(bank:dict):
    '''
    the dataset rows behind the bank's X, in order, from its stored embed.py args (held-out rows of embed.py
    --holdout come last and are left out)
    '''
    embed_args=bank["args"]
    dataset=load_images(embed_args["dataset"],embed_args["split"],embed_args["limit"],embed_args.get("data_files"),
                        embed_args.get("shuffle",False),embed_args.get("seed",123))
    return dataset.select(range(bank["X"].shape[0]))


def tokens_path(bank_path:str)->str:
    # where clip_tokens.py stores a bank's CLIP token embeddings: next to it, too big to live inside it
    return os.path.splitext(bank_path)[0]+"_tokens.pt"


def metadata_strings(ds,column:str)->list:
    '''
    a metadata column as readable strings for captions: class labels become their names, "_"/"-" become spaces
    (WikiArt's "claude-monet", "Abstract_Expressionism"), only the first of comma-separated synonyms is kept
    (ImageNet's "tench, Tinca tinca"), and "Unknown ..." placeholders become empty
    '''
    feature=ds.features[column]
    values=ds[column]
    if hasattr(feature,"names"):
        values=[feature.names[v] for v in values]
    out=[]
    for v in values:
        v=str(v or "").split(",")[0].replace("_"," ").replace("-"," ").strip()
        out.append("" if v.lower().startswith("unknown") else v)
    return out


def compose_caption(template:str,**fields)->str:
    '''
    fill a caption template like "{blip}, {style} {genre} by {artist}" and tidy what empty fields leave behind
    (double spaces, a dangling "by", empty comma-separated pieces)
    '''
    import re
    text=template.format(**fields)
    text=re.sub(r"\bby\s*(?=,|$)","",text)
    text=re.sub(r"\s+"," ",text)
    pieces=[p.strip() for p in text.split(",")]
    return ", ".join(p for p in pieces if p)
