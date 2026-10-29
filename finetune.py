'''
train a DiffusionNet (pretrained.py) on a bank: the checkpoint's own objective, DDPM epsilon prediction
    loss = || eps_theta(sqrt(abar_k) x + sqrt(1-abar_k) eps, k, c) - eps ||^2,  k ~ U{0..999}
--init pretrained: start from the checkpoint (with --cross_attention, zero-initialized adapters, so training
                   starts from the pretrained model and learns text conditioning)
--init scratch:    same architecture, random init, trained only on this bank (so the network and the
                   closed-form samplers see exactly the same images)
captions (bank["captions"], from embed.py --label_column or caption.py) are encoded to CLIP token states and
dropped with prob --cfg_dropout (empty caption) for classifier-free guidance
re-running is safe: a run whose checkpoint already reached --max_steps exits immediately, and a partial run
(e.g. a preempted, requeued job) resumes from its last save, including optimizer and LR-schedule state;
--overwrite starts over
usage: sbatch runpygpu_chip.sh finetune.py --bank banks/cifar10_pix32.pt --base google/ddpm-cifar10-32 --init pretrained --cross_attention --name cifar10_ft
'''
import os
import argparse
import time
import copy
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from experiment_helpers.gpu_details import print_details
from bank import load_clip,encode_texts_clip_tokens
from pretrained import DiffusionNet,BASES

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/cifar10_pix32.pt")
parser.add_argument("--base",type=str,default="google/ddpm-cifar10-32",choices=list(BASES))
parser.add_argument("--init",type=str,default="pretrained",choices=["pretrained","scratch"])
parser.add_argument("--sample_size",type=int,default=0,help="run the base architecture at this resolution (e.g. 64 for celebahq-256 on 64px faces); 0 keeps the base's")
parser.add_argument("--channels",type=int,default=0,help="--init scratch: input/output channels (e.g. 4 for SD-VAE latents with the ldm-celebahq-256 UNet); 0 keeps the base's")
parser.add_argument("--cross_attention",action="store_true",help="add text cross-attention adapters (needs bank captions)")
parser.add_argument("--adapters_only",action="store_true",help="with --init pretrained, freeze the UNet and train only the adapters")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--name",type=str,default="cifar10_ft")
parser.add_argument("--save_dir",type=str,default="weights")
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--warmup_steps",type=int,default=500)
parser.add_argument("--max_steps",type=int,default=50000)
parser.add_argument("--cfg_dropout",type=float,default=0.1)
parser.add_argument("--ema_decay",type=float,default=0.9999)
parser.add_argument("--save_every",type=int,default=5000)
parser.add_argument("--log_every",type=int,default=100)
parser.add_argument("--gradient_checkpointing",action="store_true",help="trade compute for memory (256px models on a 10GB GPU)")
parser.add_argument("--mixed_precision",type=str,default="bf16",choices=["no","fp16","bf16"])
parser.add_argument("--overwrite",action="store_true",help="retrain from scratch even if a checkpoint exists")
parser.add_argument("--project_name",type=str,default="notrain")
parser.add_argument("--no_wandb",action="store_true")
parser.add_argument("--seed",type=int,default=123)


def main(args):
    save_subdir=os.path.join(args.save_dir,args.name)
    meta_path=os.path.join(save_subdir,"ema.json")
    if not args.overwrite and os.path.exists(meta_path):
        with open(meta_path) as f:
            done=json.load(f).get("step",0)
        if done>=args.max_steps:
            print(f"{save_subdir} already trained to step {done} >= --max_steps {args.max_steps}; nothing to do (--overwrite to retrain)")
            return
    accelerator=Accelerator(mixed_precision=args.mixed_precision,log_with=None if args.no_wandb else "wandb")
    device=accelerator.device
    torch.manual_seed(args.seed)
    if not args.no_wandb:
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    bank=torch.load(args.bank,weights_only=False)
    n=bank["X"].shape[0] if args.limit<=0 else min(args.limit,bank["X"].shape[0])
    X=bank["X"][:n]
    shape=bank["shape"]
    print(f"training {args.init} {args.base}{' + cross-attention' if args.cross_attention else ''} on {n} bank rows {shape} ({bank['space']})")

    token_dim=768
    if args.cross_attention:
        assert "captions" in bank,"bank has no captions: embed.py --label_column or caption.py"
        captions=bank["captions"][:n]
        clip_model,clip_processor=load_clip(bank["clip_model"],device)
        clip_model.requires_grad_(False)
        token_dim=clip_model.config.text_config.hidden_size
        # few distinct captions (class labels): encode each once
        cache={c:None for c in set(captions)} if len(set(captions))<=1000 else None
        def encode(texts:list):
            if cache is None:
                return encode_texts_clip_tokens(clip_model,clip_processor,texts,device)
            for c in set(texts):
                if cache.get(c) is None:
                    cache[c]=encode_texts_clip_tokens(clip_model,clip_processor,[c],device)
            return torch.cat([cache[c][0] for c in texts]),torch.cat([cache[c][1] for c in texts])
        null_text,null_mask=encode_texts_clip_tokens(clip_model,clip_processor,[""],device)

    model=DiffusionNet(args.base,args.init,args.cross_attention,token_dim,args.sample_size or None,args.channels or None)
    assert tuple(model.unet.config.sample_size if isinstance(model.unet.config.sample_size,(list,tuple)) else [model.unet.config.sample_size]*2)==tuple(shape[1:]) and model.unet.config.in_channels==shape[0],\
        f"{args.base} expects {model.unet.config.in_channels}x{model.unet.config.sample_size} inputs but the bank is {shape}"
    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
    if args.adapters_only:
        assert args.init=="pretrained" and args.cross_attention,"--adapters_only needs --init pretrained --cross_attention"
        model.unet.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad]
    print(f"{sum(p.numel() for p in params)/1e6:.1f}M trainable parameters, {len(model.adapters)} cross-attention adapters")
    ema=copy.deepcopy(model).requires_grad_(False)

    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.0,(s+1)/args.warmup_steps))
    loader=DataLoader(torch.arange(n),batch_size=args.batch_size,shuffle=True,drop_last=n>=args.batch_size)
    model,optimizer,scheduler,loader=accelerator.prepare(model,optimizer,scheduler,loader)
    ema=ema.to(device)
    alphas_cumprod=accelerator.unwrap_model(model).alphas_cumprod.to(device)

    os.makedirs(save_subdir,exist_ok=True)
    meta={"bank":args.bank,"n_train":n,"args":vars(args)}
    state_path=os.path.join(save_subdir,"train_state.pt")

    def save(step:int):
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save(os.path.join(save_subdir,"model"),{**meta,"step":step})
            ema.save(os.path.join(save_subdir,"ema"),{**meta,"step":step})
            torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"step":step},state_path)
            print(f"saved step {step} to {save_subdir}")

    step=0
    if not args.overwrite and os.path.exists(state_path):
        # resume a partial run (e.g. preempted and requeued)
        state=torch.load(state_path,map_location=device,weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(torch.load(os.path.join(save_subdir,"model.pt"),map_location=device,weights_only=True))
        ema.load_state_dict(torch.load(os.path.join(save_subdir,"ema.pt"),map_location=device,weights_only=True))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step=state["step"]
        print(f"resuming {save_subdir} from step {step}")
    start=time.time()
    loss_buffer=[]
    while step<args.max_steps:
        for idx in loader:
            idx=idx.cpu()
            x=X[idx].to(device).float().reshape(-1,*shape)
            B=x.shape[0]
            k=torch.randint(0,len(alphas_cumprod),(B,),device=device)
            abar=alphas_cumprod[k].view(-1,1,1,1)
            eps=torch.randn_like(x)
            x_k=abar.sqrt()*x+(1-abar).sqrt()*eps
            text=mask=None
            if args.cross_attention:
                drop=(torch.rand(B)<args.cfg_dropout).tolist()
                text,mask=encode(["" if d else captions[i] for i,d in zip(idx.tolist(),drop)])
            with accelerator.autocast():
                pred=accelerator.unwrap_model(model).eps(x_k,k.float(),text,mask)
            loss=F.mse_loss(pred.float(),eps)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(params,1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            with torch.no_grad():
                for p_ema,p in zip(ema.parameters(),accelerator.unwrap_model(model).parameters()):
                    p_ema.lerp_(p,1-args.ema_decay)
            loss_buffer.append(loss.item())
            step+=1
            if step%args.log_every==0:
                mean_loss=sum(loss_buffer)/len(loss_buffer)
                accelerator.print(f"\t step {step} loss {mean_loss:.4f} elapsed {time.time()-start:.0f}s")
                accelerator.log({"loss":mean_loss,"lr":scheduler.get_last_lr()[0]},step=step)
                loss_buffer=[]
            if step%args.save_every==0:
                save(step)
            if step>=args.max_steps:
                break
    if step%args.save_every!=0:
        save(step)
    accelerator.end_training()


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")
