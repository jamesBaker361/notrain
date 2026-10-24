'''
text-conditioned UNet for flow matching on bank latents, z_t = t x + (1-t) eps, predicting v = x - eps
(so x0_hat = z + (1-t) v, directly comparable to the closed-form denoisers in els.py)

attention=False: conv-only UNet (no self- or cross-attention anywhere); the normalized pooled CLIP text
                 embedding is projected and added to the time embedding
attention=True:  UNet with self+cross-attention blocks over the per-token CLIP text states
'''
import json
import torch
import torch.nn as nn
from diffusers import UNet2DModel,UNet2DConditionModel


class TextUNet(nn.Module):
    def __init__(self,in_channels:int=4,sample_size:int=32,block_out_channels:tuple=(128,256,256,256),layers_per_block:int=2,
                 text_dim:int=768,token_dim:int=768,attention:bool=False,norm_num_groups:int=32,attention_head_dim:int=8):
        super().__init__()
        self.config={"in_channels":in_channels,"sample_size":sample_size,"block_out_channels":list(block_out_channels),
                     "layers_per_block":layers_per_block,"text_dim":text_dim,"token_dim":token_dim,"attention":attention,
                     "norm_num_groups":norm_num_groups,"attention_head_dim":attention_head_dim}
        n=len(block_out_channels)
        self.attention=attention
        if attention:
            self.unet=UNet2DConditionModel(sample_size=sample_size,in_channels=in_channels,out_channels=in_channels,
                                           block_out_channels=tuple(block_out_channels),layers_per_block=layers_per_block,
                                           down_block_types=("CrossAttnDownBlock2D",)*(n-1)+("DownBlock2D",),
                                           up_block_types=("UpBlock2D",)+("CrossAttnUpBlock2D",)*(n-1),
                                           cross_attention_dim=token_dim,norm_num_groups=norm_num_groups,
                                           attention_head_dim=attention_head_dim)
        else:
            self.unet=UNet2DModel(sample_size=sample_size,in_channels=in_channels,out_channels=in_channels,
                                  block_out_channels=tuple(block_out_channels),layers_per_block=layers_per_block,
                                  down_block_types=("DownBlock2D",)*n,up_block_types=("UpBlock2D",)*n,
                                  class_embed_type="identity",add_attention=False,norm_num_groups=norm_num_groups)
            self.text_proj=nn.Linear(text_dim,block_out_channels[0]*4)

    def forward(self,z:torch.Tensor,t:torch.Tensor,text:torch.Tensor,text_mask:torch.Tensor=None)->torch.Tensor:
        '''
        z: (B,C,H,W), t: (B,) in [0,1], text: (B,text_dim) pooled embedding, or (B,77,token_dim) token
        states with text_mask (B,77) for the attention variant. Returns the velocity v
        '''
        timesteps=t*1000
        if self.attention:
            return self.unet(z,timesteps,encoder_hidden_states=text,encoder_attention_mask=text_mask).sample
        return self.unet(z,timesteps,class_labels=self.text_proj(text)).sample

    def x0(self,z:torch.Tensor,t:torch.Tensor,text:torch.Tensor,text_mask:torch.Tensor=None)->torch.Tensor:
        return z+(1-t).view(-1,1,1,1)*self(z,t,text,text_mask)

    def save(self,path:str,extra:dict=None):
        torch.save(self.state_dict(),path+".pt")
        with open(path+".json","w") as f:
            json.dump({"model":self.config,**(extra or {})},f,indent=4)

    @classmethod
    def load(cls,path:str,map_location="cpu"):
        with open(path+".json") as f:
            meta=json.load(f)
        model=cls(**meta["model"])
        model.load_state_dict(torch.load(path+".pt",map_location=map_location,weights_only=True))
        return model,meta
