'''
train a text-conditioned flow-matching UNet (textunet.py) on the latents of a captioned bank
(embed.py then caption.py), so the network and the closed-form machines in els.py share one training set
loss: || v_theta(z_t, t, c) - (x - eps) ||^2 with z_t = t x + (1-t) eps, t ~ U(0,1)
captions are dropped (replaced by the empty caption) with prob --cfg_dropout for classifier-free guidance
--unconditional trains without text (no captions needed), e.g. the trained-DDPM baseline for the pixel tree
usage: sbatch runpygpu_chip.sh train.py --bank banks/celeb.pt --name celeb_conv
       sbatch runpygpu_chip.sh train.py --bank banks/celeb.pt --name celeb_attn --attention
'''
import os
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from experiment_helpers.gpu_details import print_details
from bank import load_clip,encode_texts_clip,encode_texts_clip_tokens
from textunet import TextUNet

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celeb.pt")
parser.add_argument("--limit",type=int,default=-1,help="train on the first n bank rows only (compare.py uses the same rows)")
parser.add_argument("--name",type=str,default="celeb_conv")
parser.add_argument("--save_dir",type=str,default="weights")
parser.add_argument("--unconditional",action="store_true",help="no text conditioning (conv-only UNet, captions not needed)")
parser.add_argument("--attention",action="store_true",help="self+cross-attention UNet over CLIP token states instead of the conv-only UNet")
parser.add_argument("--block_out_channels",type=int,nargs="+",default=[128,256,256,256])
parser.add_argument("--layers_per_block",type=int,default=2)
parser.add_argument("--norm_num_groups",type=int,default=32)
parser.add_argument("--batch_size",type=int,default=128)
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--warmup_steps",type=int,default=1000)
parser.add_argument("--max_steps",type=int,default=100000)
parser.add_argument("--cfg_dropout",type=float,default=0.1)
parser.add_argument("--ema_decay",type=float,default=0.9999)
parser.add_argument("--save_every",type=int,default=5000)
parser.add_argument("--log_every",type=int,default=100)
parser.add_argument("--mixed_precision",type=str,default="bf16",choices=["no","fp16","bf16"])
parser.add_argument("--project_name",type=str,default="notrain")
parser.add_argument("--no_wandb",action="store_true")
parser.add_argument("--seed",type=int,default=123)


def main(args):
    accelerator=Accelerator(mixed_precision=args.mixed_precision,log_with=None if args.no_wandb else "wandb")
    device=accelerator.device
    torch.manual_seed(args.seed)
    if not args.no_wandb:
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    bank=torch.load(args.bank,weights_only=False)
    n=bank["X"].shape[0] if args.limit<=0 else min(args.limit,bank["X"].shape[0])
    X=bank["X"][:n].reshape(n,*bank["shape"])
    if args.unconditional:
        assert not args.attention,"--unconditional uses the conv-only UNet"
        # a constant zero "text" input: the text projection reduces to a learned bias
        C=torch.zeros(n,1)
        token_dim=1
        print(f"training unconditionally on {n} images {tuple(X.shape[1:])}")
    else:
        assert "C" in bank and "captions" in bank,"bank has no captions, run caption.py first"
        C=bank["C"][:n]
        captions=bank["captions"][:n]
        print(f"training on {n} images {tuple(X.shape[1:])}, e.g. '{captions[0]}'")
        clip_model,clip_processor=load_clip(bank["clip_model"],device)
        clip_model.requires_grad_(False)
        null_pooled=encode_texts_clip(clip_model,clip_processor,[""],device)[0]
        token_dim=clip_model.config.text_config.hidden_size

    model=TextUNet(in_channels=bank["shape"][0],sample_size=bank["shape"][1],block_out_channels=tuple(args.block_out_channels),
                   layers_per_block=args.layers_per_block,text_dim=C.shape[1],token_dim=token_dim,
                   attention=args.attention,norm_num_groups=args.norm_num_groups)
    ema=copy.deepcopy(model).requires_grad_(False)
    print(f"{'attention' if args.attention else 'conv-only'} UNet with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.0,(s+1)/args.warmup_steps))
    loader=DataLoader(torch.arange(n),batch_size=args.batch_size,shuffle=True,drop_last=n>=args.batch_size)
    model,optimizer,scheduler,loader=accelerator.prepare(model,optimizer,scheduler,loader)
    ema=ema.to(device)

    save_subdir=os.path.join(args.save_dir,args.name)
    os.makedirs(save_subdir,exist_ok=True)
    meta={"bank":args.bank,"n_train":n,"attention":args.attention,"unconditional":args.unconditional,"args":vars(args)}

    def save(step:int):
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save(os.path.join(save_subdir,"model"),{**meta,"step":step})
            ema.save(os.path.join(save_subdir,"ema"),{**meta,"step":step})
            print(f"saved step {step} to {save_subdir}")

    step=0
    start=time.time()
    loss_buffer=[]
    while step<args.max_steps:
        for idx in loader:
            idx=idx.cpu()
            x=X[idx].to(device)
            B=x.shape[0]
            drop=torch.rand(B)<args.cfg_dropout
            if args.unconditional:
                text=C[idx].to(device)
                text_mask=None
            elif args.attention:
                texts=["" if d else captions[i] for i,d in zip(idx.tolist(),drop.tolist())]
                text,text_mask=encode_texts_clip_tokens(clip_model,clip_processor,texts,device)
            else:
                text=C[idx].to(device)
                text[drop.to(device)]=null_pooled
                text_mask=None
            t=torch.rand(B,device=device)
            eps=torch.randn_like(x)
            z=t.view(-1,1,1,1)*x+(1-t.view(-1,1,1,1))*eps
            with accelerator.autocast():
                v=model(z,t,text,text_mask)
            loss=F.mse_loss(v.float(),x-eps)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(),1.0)
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
