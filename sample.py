'''
sample text-conditioned smoothed CFDM from a bank built by embed.py
usage: sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompt_file prompts/faces.txt --w 0 1 4
   or: sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompts "a smiling woman" "a man with a beard" --w 0 1 4
each output row is: [top CLIP-retrieved training image | samples...] for one (prompt, w)
--scales 1 2 4 turns on multiscale patches: whole image, quarters and 16ths, blended over time by --schedule
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
from cfdm import text_log_prior,top_k_subset,sample,sample_multiscale,median_patch_distance
from bank import load_clip,encode_texts_clip,encode_images_clip,load_vae,bank_to_pixels,to_pil
from tree import tree_sample,node_size
from locality import global_spectrum

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celeb_vae_256.pt")
parser.add_argument("--prompt_file",type=str,default="prompts/faces.txt",help="one prompt per line; blank lines are skipped")
parser.add_argument("--prompts",type=str,nargs="+",default=None,help="overrides --prompt_file")
parser.add_argument("--n_per_prompt",type=int,default=8)
parser.add_argument("--w",type=float,nargs="+",default=[0.0,1.0,4.0],help="text guidance weights to sweep; 0=unconditional, 1=exact conditional")
parser.add_argument("--sigma",type=float,default=0.25,help="smoothing, as a fraction of the median pairwise distance between bank points; 0 reproduces training points exactly")
parser.add_argument("--M",type=int,default=4,help="noise samples per smoothed score evaluation")
parser.add_argument("--steps",type=int,default=50)
parser.add_argument("--start_t",type=float,default=0.2,help="start from the exact mixture at this t (0 = pure noise)")
parser.add_argument("--top_k",type=int,default=1000,help="restrict to the top-k CLIP matches per prompt (<=0 for all); w=0 always uses the full bank")
parser.add_argument("--scales",type=int,nargs="+",default=[1],help="patch grid sizes, coarse to fine, e.g. 1 2 4 = whole, quarters, 16ths")
parser.add_argument("--patch_sigma",type=float,default=0.0,help="smoothing at the patch scales (>1), as a fraction of the median patch distance at that scale; --sigma applies to the whole-image scale")
parser.add_argument("--patch_positions",type=str,default="any",choices=["any","same"],help="whether a region can draw patches from every grid position or only its own")
parser.add_argument("--schedule",type=str,default="coarse_to_fine",choices=["coarse_to_fine","fine_to_coarse","constant"])
parser.add_argument("--scale_weights",type=float,nargs="+",default=None,help="per-scale weights for --schedule constant")
parser.add_argument("--method",type=str,default="cfdm",choices=["cfdm","tree"],help="tree: descend a CLIP tree built by tree.py, then sample from the node")
parser.add_argument("--tree",type=str,default="",help="tree built by tree.py for this bank")
parser.add_argument("--tree_depth",type=int,default=5,help="stop the descent at this depth (L)")
parser.add_argument("--c_search",type=int,default=20,help="stop the descent at the first node with this many images or fewer")
parser.add_argument("--node_sampler",type=str,default="lowrank",choices=["diagonal","lowrank","cfdm","locality","wiener"],help="how to sample from the stopping node; cfdm uses --sigma, --M, --steps, --start_t; locality/wiener (Lukoianov et al.) use --steps, --tau, --wiener_shrinkage, --wiener_ridge")
parser.add_argument("--tau",type=float,default=0.02,help="locality: Wiener-matrix binarization threshold, relative to each row's max")
parser.add_argument("--wiener_shrinkage",type=float,default=0.0,help="locality/wiener: blend the node covariance toward the full-bank covariance (0 = node only, 1 = global)")
parser.add_argument("--wiener_ridge",type=float,default=0.01,help="locality/wiener: add this fraction of the mean pixel variance to the covariance diagonal so small (rank-deficient) nodes still get local masks at low noise")
parser.add_argument("--wiener_exact_eig",action="store_true",help="with --wiener_shrinkage, eigendecompose each node's blended covariance instead of the (equivalent, faster) Woodbury update of the global one")
parser.add_argument("--node_text_prior",action="store_true",help="locality: also weight node images by w * log p(i|c) inside the softmax")
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

    if args.prompts is not None:
        prompts=args.prompts
    else:
        with open(args.prompt_file,"r") as file:
            prompts=[line.strip() for line in file if line.strip()]
    print("prompts",prompts)

    if not args.no_wandb:
        wandb.init(project=args.project_name,config=vars(args))

    clip_model,clip_processor=load_clip(bank["clip_model"],device)
    vae=load_vae(bank["vae"],device) if bank["space"]=="vae" else None
    text_embeds=encode_texts_clip(clip_model,clip_processor,prompts,device)
    log_priors=text_log_prior(text_embeds,T,bank["logit_scale"])
    if args.eval_clip_model:
        eval_model,eval_processor=load_clip(args.eval_clip_model,device)
        eval_text_embeds=encode_texts_clip(eval_model,eval_processor,prompts,device)
    else:
        eval_model,eval_processor,eval_text_embeds=clip_model,clip_processor,text_embeds

    if args.method=="tree":
        assert args.tree,"--method tree needs --tree"
        tree=torch.load(args.tree,weights_only=False)
        assert tree["n"]==X.shape[0],f"tree was built for {tree['n']} images but the bank has {X.shape[0]}"
        eig_cache={}
        S_global=None
        spectrum=None
        if args.node_sampler in ["locality","wiener"] and args.wiener_shrinkage>0:
            # global covariance without materializing a centered copy of the bank
            mu=X.mean(0)
            S_global=(X.T@X-X.shape[0]*torch.outer(mu,mu))/(X.shape[0]-1)
            if not args.wiener_exact_eig:
                # Woodbury path: one eigendecomposition for the run instead of one per node
                spectrum=global_spectrum(S_global)
                S_global=None

    if args.schedule=="constant":
        assert args.scale_weights is not None and len(args.scale_weights)==len(args.scales),"--schedule constant needs one --scale_weights per scale"

    # typical distance between bank points (and between patches at each scale) over the whole bank:
    # sets the smoothing per scale; the scale-1 value also normalizes the copy metric
    generator=torch.Generator(device=device).manual_seed(args.seed)
    scale=median_patch_distance(X,bank["shape"],1,generator=generator)
    sigmas=[(args.sigma if s==1 else args.patch_sigma)*median_patch_distance(X,bank["shape"],s,generator=generator) for s in args.scales]
    sigma=args.sigma*scale
    print(f"median pairwise distance {scale:.4f}, sigma {sigma:.4f}, per-scale sigmas {dict(zip(args.scales,sigmas))}")

    os.makedirs(args.output_dir,exist_ok=True)
    results=[]
    for p,prompt in enumerate(prompts):
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
            nodes=None
            if args.method=="tree":
                # w sets the descent sharpness (0 = uniform random walk), so no top-k subset is used
                prior_mass=None
                out,nodes=tree_sample(tree,X,text_embeds[p],args.n_per_prompt,w,bank["logit_scale"],args.tree_depth,
                                      args.c_search,args.node_sampler,generator,
                                      cfdm_kwargs={"log_prior":log_priors[p],"w":w,"sigma":sigma,"M":args.M,
                                                   "steps":args.steps,"start_t":args.start_t},
                                      locality_kwargs={"tau":args.tau,"steps":args.steps,"shrinkage":args.wiener_shrinkage,"ridge":args.wiener_ridge,
                                                       "S_global":S_global,"spectrum":spectrum,"cache":eig_cache,
                                                       "log_prior":log_priors[p] if args.node_text_prior else None,"w":w})
            elif args.scales==[1]:
                out=sample(args.n_per_prompt,X_w,log_prior_w,w=w,sigma=sigma,M=args.M,
                           steps=args.steps,start_t=args.start_t,generator=generator)
            else:
                out=sample_multiscale(args.n_per_prompt,X_w,bank["shape"],args.scales,log_prior_w,w=w,sigmas=sigmas,M=args.M,
                                      steps=args.steps,start_t=args.start_t,schedule=args.schedule,
                                      constant_weights=args.scale_weights,positions=args.patch_positions,generator=generator)
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
            if nodes is not None:
                metrics["node_size"]=float(np.mean([node_size(tree,m) for m in nodes]))
                metrics["distinct_nodes"]=len(set(nodes))
            print(metrics)
            results.append(metrics)
            row=concat_images_horizontally([retrieved]+images)
            rows.append(row)
            if not args.no_wandb:
                wandb.log({f"{name}/w={w}":wandb.Image(row),**{f"{name}/{k}":v for k,v in metrics.items() if k!="prompt"}})
        grid=concat_images_vertically(rows)
        grid.save(os.path.join(args.output_dir,f"{p}_{name}.png"))

    with open(os.path.join(args.output_dir,"metrics.json"),"w") as f:
        json.dump({"args":vars(args),"prompts":prompts,"results":results},f,indent=4)
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
