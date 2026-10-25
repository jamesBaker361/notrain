'''
trained diffusion baselines built on diffusers DDPM / latent-diffusion UNets:
    google/ddpm-cifar10-32       pixel CIFAR-10 32x32
    google/ddpm-celebahq-256     pixel CelebA-HQ 256x256
    CompVis/ldm-celebahq-256     CelebA-HQ in the 3x64x64 VQ latent space of its vqvae (bank space "vq")
all are UNet2DModel epsilon-predictors with a discrete 1000-step schedule (alphas_cumprod).

DiffusionNet runs them in this repo's convention z_t = t x + (1-t) eps (t=0 noise, t=1 data), so they share the
time grid and initial noise of the closed-form samplers. With n = sqrt(t^2 + (1-t)^2):
    x_k = z / n is the DDPM state with sqrt(abar) = t / n and sqrt(1 - abar) = (1-t) / n,
    the (continuous) timestep k is interpolated from the schedule in log-SNR, and
    x0_hat = (x_k - sqrt(1 - abar) eps_hat) / sqrt(abar)    (clipped to [-1,1] for pixel models, like DDPMPipeline)

cross_attention=True adds a CrossAttnAdapter after every self-attention layer of the UNet (via forward hooks):
group norm -> multi-head attention from the feature map to CLIP text token states -> zero-initialized output
projection, added residually. With zero init the model starts exactly as the pretrained one, and fine-tuning
(finetune.py) learns text conditioning. Text dropout during training makes the empty caption the unconditional
model, so classifier-free guidance uses the same w as the rest of the code: eps = eps_u + w (eps_c - eps_u),
w=0 unconditional, w=1 conditional.

init="scratch" keeps the architecture but re-initializes the weights (trained on our bank by finetune.py).
'''
import json
import math
import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DModel,DDPMScheduler
from diffusers.models.attention_processor import Attention

BASES={
    "google/ddpm-cifar10-32":{"unet_subfolder":None,"scheduler_subfolder":None,"clip_x0":True},
    "google/ddpm-celebahq-256":{"unet_subfolder":None,"scheduler_subfolder":None,"clip_x0":True},
    "CompVis/ldm-celebahq-256":{"unet_subfolder":"unet","scheduler_subfolder":"scheduler","clip_x0":False},
}


def _heads(channels:int)->int:
    # ~64-dim heads that divide the channel count
    h=max(1,channels//64)
    while channels%h:
        h-=1
    return h


class CrossAttnAdapter(nn.Module):
    def __init__(self,channels:int,token_dim:int):
        super().__init__()
        self.norm=nn.GroupNorm(min(32,channels),channels)
        self.attn=nn.MultiheadAttention(channels,_heads(channels),kdim=token_dim,vdim=token_dim,batch_first=True)
        self.out=nn.Linear(channels,channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self,h:torch.Tensor,text:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        B,C,H,W=h.shape
        q=self.norm(h).flatten(2).transpose(1,2)
        a,_=self.attn(q,text.to(q.dtype),text.to(q.dtype),key_padding_mask=None if mask is None else ~mask.bool(),need_weights=False)
        return h+self.out(a).transpose(1,2).reshape(B,C,H,W)


class DiffusionNet(nn.Module):
    def __init__(self,base:str,init:str="pretrained",cross_attention:bool=False,token_dim:int=768):
        super().__init__()
        assert base in BASES,f"unknown base {base}, expected one of {list(BASES)}"
        spec=BASES[base]
        self.config={"base":base,"init":init,"cross_attention":cross_attention,"token_dim":token_dim}
        kwargs={"subfolder":spec["unet_subfolder"]} if spec["unet_subfolder"] else {}
        if init=="pretrained":
            self.unet=UNet2DModel.from_pretrained(base,**kwargs)
        else:
            self.unet=UNet2DModel.from_config(UNet2DModel.load_config(base,**kwargs))
        skw={"subfolder":spec["scheduler_subfolder"]} if spec["scheduler_subfolder"] else {}
        # the training noise schedule (DDPM betas) of the checkpoint, whatever sampler it shipped with
        scheduler=DDPMScheduler.from_config(DDPMScheduler.load_config(base,**skw))
        self.register_buffer("alphas_cumprod",scheduler.alphas_cumprod.float().clone(),persistent=False)
        self.clip_x0=spec["clip_x0"]
        self.adapters=nn.ModuleList()
        self._text=None
        self._mask=None
        if cross_attention:
            for module in self.unet.modules():
                if isinstance(module,Attention):
                    adapter=CrossAttnAdapter(module.to_q.in_features,token_dim)
                    self.adapters.append(adapter)
                    module.register_forward_hook(self._hook(adapter))

    def _hook(self,adapter:CrossAttnAdapter):
        def hook(module,inputs,output):
            if self._text is None:
                return output
            return adapter(output,self._text,self._mask)
        return hook

    @property
    def t_min(self)->float:
        '''
        the noisiest t the checkpoint was trained on (its last timestep); below it x0_hat divides by a tiny
        sqrt(abar), so comparisons should start every method here
        '''
        a=math.sqrt(float(self.alphas_cumprod[-1]))
        return a/(a+math.sqrt(1-a*a))

    def timestep(self,abar:torch.Tensor)->torch.Tensor:
        # continuous timestep k with alphas_cumprod[k] = abar, interpolated in log-SNR (decreasing in k)
        logsnr=torch.log(self.alphas_cumprod/(1-self.alphas_cumprod)).cpu().numpy()
        target=torch.log(abar/(1-abar)).clamp(float(logsnr[-1]),float(logsnr[0])).cpu().numpy()
        k=np.interp(-target,-logsnr,np.arange(len(logsnr)))
        return torch.tensor(k,dtype=torch.float32,device=abar.device)

    def eps(self,x_k:torch.Tensor,k:torch.Tensor,text:torch.Tensor=None,mask:torch.Tensor=None)->torch.Tensor:
        self._text,self._mask=text,mask
        try:
            return self.unet(x_k,k).sample
        finally:
            self._text,self._mask=None,None

    def x0(self,z:torch.Tensor,t:float,text:torch.Tensor=None,mask:torch.Tensor=None,
           null_text:torch.Tensor=None,null_mask:torch.Tensor=None,w:float=1.0)->torch.Tensor:
        '''
        x0_hat in this repo's convention from z_t (B,C,H,W). With text: CFG with weight w against null_text
        (w=0 uses only the unconditional prediction, w=1 only the conditional one)
        '''
        B=z.shape[0]
        n=math.sqrt(t*t+(1-t)**2)
        a,b=t/n,(1-t)/n
        x_k=z/n
        k=self.timestep(torch.full((B,),a*a,device=z.device)).clamp(0,len(self.alphas_cumprod)-1)
        if text is None or not len(self.adapters):
            e=self.eps(x_k,k)
        elif w==0:
            e=self.eps(x_k,k,null_text,null_mask)
        elif w==1:
            e=self.eps(x_k,k,text,mask)
        else:
            e_u=self.eps(x_k,k,null_text,null_mask)
            e=e_u+w*(self.eps(x_k,k,text,mask)-e_u)
        x0=(x_k-b*e)/a
        return x0.clamp(-1,1) if self.clip_x0 else x0

    def save(self,path:str,extra:dict=None):
        torch.save(self.state_dict(),path+".pt")
        with open(path+".json","w") as f:
            json.dump({"diffusion_net":self.config,**(extra or {})},f,indent=4)

    @classmethod
    def load(cls,spec:str,map_location="cpu"):
        '''
        spec "pretrained:<repo>" loads the checkpoint as-is (no cross-attention); otherwise a path prefix
        written by finetune.py
        '''
        if spec.startswith("pretrained:"):
            return cls(spec.split(":",1)[1],"pretrained",False),{"spec":spec}
        with open(spec+".json") as f:
            meta=json.load(f)
        model=cls(**{**meta["diffusion_net"],"init":"scratch"})
        model.load_state_dict(torch.load(spec+".pt",map_location=map_location,weights_only=True))
        return model,meta


@torch.no_grad()
def diffusion_sample(model:DiffusionNet,z0:torch.Tensor,steps:int=10,t_min:float=1e-3,text:torch.Tensor=None,mask:torch.Tensor=None,
                     null_text:torch.Tensor=None,null_mask:torch.Tensor=None,w:float=1.0)->torch.Tensor:
    '''
    Euler on v = (x0_hat - z)/(1-t) from z0 at t_min to t = 1, on the closed-form samplers' time grid
    (for this interpolation it is the deterministic DDIM update); the last step lands on x0_hat
    '''
    ts=torch.linspace(t_min,1.0,steps+1).tolist()
    z=z0
    for t,t_next in zip(ts[:-1],ts[1:]):
        x0=model.x0(z,t,text,mask,null_text,null_mask,w)
        z=z+(t_next-t)*(x0-z)/(1-t)
    return z


class TrainedNet:
    '''
    one interface for every trained baseline: DiffusionNet specs ("pretrained:<repo>" or a finetune.py prefix)
    and train.py TextUNet prefixes. sample(z0, steps, t_min, prompt, w) -> (B,C,H,W); prompts need clip_fn
    (texts -> (states, mask)) and are ignored by unconditional models
    '''
    def __init__(self,spec:str,device,clip_fn=None):
        self.device=device
        self.clip_fn=clip_fn
        if spec.startswith("pretrained:") or self._is_diffusion_net(spec):
            self.kind="diffusion"
            self.model,self.meta=DiffusionNet.load(spec,map_location=device)
        else:
            from textunet import TextUNet
            self.kind="textunet"
            self.model,self.meta=TextUNet.load(spec,map_location=device)
            assert self.meta.get("unconditional"),"TextUNet baselines must be --unconditional"
        self.model=self.model.to(device).eval()
        self.conditional=self.kind=="diffusion" and len(self.model.adapters)>0
        # where sampling should start: the checkpoint's noisiest training point (TextUNet trains on all of [0,1])
        self.t_min=self.model.t_min if self.kind=="diffusion" else 1e-3
        self._null=None

    @staticmethod
    def _is_diffusion_net(spec:str)->bool:
        try:
            with open(spec+".json") as f:
                return "diffusion_net" in json.load(f)
        except FileNotFoundError:
            return False

    @torch.no_grad()
    def sample(self,z0:torch.Tensor,steps:int=10,t_min:float=1e-3,prompt:str=None,w:float=0.0)->torch.Tensor:
        if self.kind=="textunet":
            from textunet import net_sample
            return net_sample(self.model,z0,steps,t_min)
        if not self.conditional:
            return diffusion_sample(self.model,z0,steps,t_min)
        B=z0.shape[0]
        if self._null is None:
            self._null=self.clip_fn([""])
        null_text,null_mask=self._null
        if prompt is None:
            # unconditional = the empty caption the model was trained with (not the adapter-free UNet,
            # whose weights fine-tuning has changed)
            prompt,w="",0.0
        text,mask=self.clip_fn([prompt])
        return diffusion_sample(self.model,z0,steps,t_min,text.expand(B,-1,-1),mask.expand(B,-1),
                                null_text.expand(B,-1,-1),null_mask.expand(B,-1),w)
