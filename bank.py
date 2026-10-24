'''
helpers for building and decoding the "bank": training points X (flattened pixels or VAE latents)
and their CLIP image embeddings T
'''
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPModel, AutoProcessor
from diffusers import AutoencoderKL


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


def load_vae(vae_name:str,device,torch_dtype=torch.float32)->AutoencoderKL:
    return AutoencoderKL.from_pretrained(vae_name,torch_dtype=torch_dtype).to(device).eval()


@torch.no_grad()
def encode_vae(vae:AutoencoderKL,pixels:torch.Tensor)->torch.Tensor:
    # pixels in [-1,1]; use the posterior mean so the bank is deterministic
    latents=vae.encode(pixels.to(vae.dtype)).latent_dist.mean
    return latents.float()*vae.config.scaling_factor


@torch.no_grad()
def decode_vae(vae:AutoencoderKL,latents:torch.Tensor)->torch.Tensor:
    return vae.decode((latents/vae.config.scaling_factor).to(vae.dtype)).sample.float()


def to_pil(pixels:torch.Tensor)->list:
    # (B,C,H,W) in [-1,1]
    arr=((pixels.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
    return [Image.fromarray(a) for a in arr]


def bank_to_pixels(bank:dict,flat:torch.Tensor,vae:AutoencoderKL=None,batch_size:int=8)->torch.Tensor:
    '''
    flat: (B,D) points in bank space -> (B,C,H,W) pixels in [-1,1]
    '''
    x=flat.reshape(-1,*bank["shape"])
    if bank["space"]=="pixel":
        return x
    return torch.cat([decode_vae(vae,x[i:i+batch_size].to(vae.device)) for i in range(0,len(x),batch_size)]).cpu()
