'''
helpers for building and decoding the "bank": training points X (flattened pixels, SD-VAE latents, or the
VQ latents of CompVis/ldm-celebahq-256) and their CLIP image embeddings T
'''
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
    vqvae subfolder of a latent-diffusion repo (e.g. CompVis/ldm-celebahq-256)
    '''
    if space=="vq":
        return VQModel.from_pretrained(vae_name,subfolder="vqvae",torch_dtype=torch_dtype).to(device).eval()
    return AutoencoderKL.from_pretrained(vae_name,torch_dtype=torch_dtype).to(device).eval()


def load_bank_vae(bank:dict,device,torch_dtype=torch.float32):
    # the autoencoder a bank was built with, or None for a pixel bank
    if bank["space"]=="pixel":
        return None
    return load_vae(bank["vae"],device,torch_dtype,bank["space"])


@torch.no_grad()
def encode_vae(vae,pixels:torch.Tensor)->torch.Tensor:
    # pixels in [-1,1]
    if isinstance(vae,VQModel):
        # pre-quantization latents, the space latent diffusion samples in (LDMPipeline decodes them unscaled)
        return vae.encode(pixels.to(vae.dtype)).latents.float()
    # posterior mean so the bank is deterministic
    latents=vae.encode(pixels.to(vae.dtype)).latent_dist.mean
    return latents.float()*vae.config.scaling_factor


@torch.no_grad()
def decode_vae(vae,latents:torch.Tensor)->torch.Tensor:
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
