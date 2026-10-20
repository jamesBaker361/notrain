'''
sample text-conditioned smoothed CFDM from a bank built by embed.py
usage: sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompts "a smiling woman" "a man with a beard" --w 0 1 4
each output row is: [top CLIP-retrieved training image | samples...] for one (prompt, w)
'''
import os
import re
import argparse
import time
import json
import torch
import wandb
import numpy as np
from experiment_helpers.gpu_details import print_details
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
from cfdm import text_log_prior,top_k_subset,sample
from bank import load_clip,encode_texts_clip,encode_images_clip,load_vae,bank_to_pixels,to_pil

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celeb_vae_256.pt")
parser.add_argument("--prompts",type=str,nargs="+",default=["a smiling woman with blonde hair","a man with a beard and glasses"])
parser.add_argument("--n_per_prompt",type=int,default=8)
parser.add_argument("--w",type=float,nargs="+",default=[0.0,1.0,4.0],help="text guidance weights to sweep; 0=unconditional, 1=exact conditional")
parser.add_argument("--sigma",type=float,default=0.25,help="smoothing, as a fraction of the median pairwise distance between bank points; 0 reproduces training points exactly")
parser.add_argument("--M",type=int,default=4,help="noise samples per smoothed score evaluation")
parser.add_argument("--steps",type=int,default=50)
parser.add_argument("--start_t",type=float,default=0.2,help="start from the exact mixture at this t (0 = pure noise)")
parser.add_argument("--top_k",type=int,default=1000,help="restrict to the top-k CLIP matches per prompt (<=0 for all); w=0 always uses the full bank")
parser.add_argument("--copy_threshold",type=float,default=1e-2,help="relative distance below which a sample counts as a training copy")
parser.add_argument("--eval_clip_model",type=str,default="",help="CLIP used for the clip_score metric; defaults to the conditioning CLIP (biased)")
parser.add_argument("--seed",type=int,default=123)
parser.add_argument("--output_dir",type=str,default="outputs")
parser.add_argument("--project_name",type=str,default="notrain")
parser.add_argument("--no_wandb",action="store_true")


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    bank=torch.load(args.bank,weights_only=False)
    X=bank["X"].to(device)
    T=bank["T"].to(device)
    print(f"bank {args.bank}: X {tuple(X.shape)} T {tuple(T.shape)} space {bank['space']} shape {bank['shape']}")

    if not args.no_wandb:
        wandb.init(project=args.project_name,config=vars(args))

    clip_model,clip_processor=load_clip(bank["clip_model"],device)
    vae=load_vae(bank["vae"],device) if bank["space"]=="vae" else None
    text_embeds=encode_texts_clip(clip_model,clip_processor,args.prompts,device)
    log_priors=text_log_prior(text_embeds,T,bank["logit_scale"])
    if args.eval_clip_model:
        eval_model,eval_processor=load_clip(args.eval_clip_model,device)
        eval_text_embeds=encode_texts_clip(eval_model,eval_processor,args.prompts,device)
    else:
        eval_model,eval_processor,eval_text_embeds=clip_model,clip_processor,text_embeds

    # typical distance between training points over the whole bank: sets the smoothing scale and the copy metric
    generator=torch.Generator(device=device).manual_seed(args.seed)
    probe=X[torch.randperm(X.shape[0],generator=generator,device=device)[:512]]
    pair=torch.cdist(probe,probe)
    scale=pair[pair>0].median().item()
    sigma=args.sigma*scale
    print(f"median pairwise distance {scale:.4f}, sigma {sigma:.4f}")

    os.makedirs(args.output_dir,exist_ok=True)
    results=[]
    for p,prompt in enumerate(args.prompts):
        name=re.sub(r"[^A-Za-z0-9]+","_",prompt).strip("_")[:60]
        X_sub,log_prior,idx=top_k_subset(X,log_priors[p],args.top_k)
        retrieved=to_pil(bank_to_pixels(bank,X[idx[:1]],vae))[0]
        rows=[]
        for w in args.w:
            if w==0:
                # unconditional baseline: the top-k subset would itself be text conditioning
                X_w,log_prior_w,prior_mass=X,None,1.0
            else:
                # fraction of the w-sharpened prior that the top-k subset keeps
                X_w,log_prior_w=X_sub,log_prior
                prior_mass=torch.softmax(w*log_priors[p],dim=-1)[idx].sum().item()
                if prior_mass<0.9:
                    print(f"warning: top_k={args.top_k} keeps only {prior_mass:.3f} of the prior mass for '{prompt}' at w={w}")
            start=time.time()
            out=sample(args.n_per_prompt,X_w,log_prior_w,w=w,sigma=sigma,M=args.M,
                       steps=args.steps,start_t=args.start_t,generator=generator)
            if device=="cuda":
                torch.cuda.synchronize()
            elapsed=time.time()-start
            nearest=torch.cdist(out,X).min(-1).values/scale
            images=to_pil(bank_to_pixels(bank,out,vae))
            clip_scores=(encode_images_clip(eval_model,eval_processor,images,device)@eval_text_embeds[p]).cpu().numpy()
            metrics={
                "prompt":prompt,
                "w":w,
                "prior_mass":prior_mass,
                "clip_score":float(np.mean(clip_scores)),
                "copy_rate":float((nearest<args.copy_threshold).float().mean()),
                "nn_dist":float(nearest.mean()),
                "seconds_per_sample":elapsed/args.n_per_prompt
            }
            print(metrics)
            results.append(metrics)
            row=concat_images_horizontally([retrieved]+images)
            rows.append(row)
            if not args.no_wandb:
                wandb.log({f"{name}/w={w}":wandb.Image(row),**{f"{name}/{k}":v for k,v in metrics.items() if k!="prompt"}})
        grid=concat_images_vertically(rows)
        grid.save(os.path.join(args.output_dir,f"{p}_{name}.png"))

    with open(os.path.join(args.output_dir,"metrics.json"),"w") as f:
        json.dump({"args":vars(args),"results":results},f,indent=4)
    print(f"saved grids and metrics to {args.output_dir}")
    if not args.no_wandb:
        wandb.finish()


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
