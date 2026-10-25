'''
how well do the analytical samplers reproduce a trained diffusion model? (Lukoianov et al. Table 1)
every method starts from the same noise z0 and runs the same time grid, from the target network's t_min (its
noisiest training point) to t=1; we report per-sample r^2 and MSE of each final sample against the target's,
mean and std over --n samples. Other trained networks (--other_models, e.g. the same architecture trained from
another init) give the ceiling ("another DDPM"). Conditional networks are run unconditionally (empty caption).
trained networks: 'pretrained:<repo>' (as-is), finetune.py prefixes, or --unconditional train.py prefixes
methods, all over the whole training set (depth 0, the paper's setting):
    ideal     exact optimal denoiser (memorizes)
    cfdm      smoothed closed-form diffusion (--sigma, --M); needs the bank on the GPU
    wiener    linear Wiener filter with the global covariance; needs d <= 20000
    locality  Wiener-mask analytical model (Lukoianov et al.); needs d <= 20000
    ls        Kamb & Ganguli local score machine with the patch schedule --ls_patch_fracs
usage: sbatch runpygpu_chip.sh agreement.py --bank banks/cifar10_pix32.pt --model weights/cifar10_scratch/ema --other_models pretrained:google/ddpm-cifar10-32 --output_dir outputs/agreement_cifar10
'''
import os
import json
import time
import argparse
import numpy as np
import torch
from experiment_helpers.gpu_details import print_details
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
from bank import bank_to_pixels,to_pil,load_bank_vae,load_clip,encode_texts_clip_tokens
from pretrained import TrainedNet
from cfdm import sample,median_patch_distance
from locality import global_spectrum,locality_sample_shrunk
from els import machine_sample,r2,patch_schedule_from_fracs,CELEBA_PATCH_FRACS

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/cifar10_pix32.pt")
parser.add_argument("--model",type=str,required=True,help="target trained network")
parser.add_argument("--other_models",type=str,nargs="*",default=[],help="other trained networks, compared to the target like the analytical methods")
parser.add_argument("--methods",type=str,nargs="+",default=["ideal","cfdm","wiener","locality","ls"],choices=["ideal","cfdm","wiener","locality","ls"])
parser.add_argument("--n",type=int,default=64,help="samples (shared noise draws)")
parser.add_argument("--batch",type=int,default=16)
parser.add_argument("--steps",type=int,default=10)
parser.add_argument("--sigma",type=float,default=0.25,help="cfdm smoothing, as a fraction of the median pairwise distance")
parser.add_argument("--M",type=int,default=4)
parser.add_argument("--tau",type=float,default=0.02)
parser.add_argument("--ls_patch_fracs",type=float,nargs="+",default=None)
parser.add_argument("--ls_chunk",type=int,default=1024,help="training images per chunk in the ls machine (memory)")
parser.add_argument("--bank_device",type=str,default="auto",choices=["auto","cpu","cuda"],help="auto keeps banks over 4GB in CPU memory (ideal and ls stream it)")
parser.add_argument("--n_show",type=int,default=8,help="samples shown in the comparison grid")
parser.add_argument("--output_dir",type=str,default="outputs/agreement")
parser.add_argument("--seed",type=int,default=123)


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    bank=torch.load(args.bank,weights_only=False)
    shape=bank["shape"]
    bank_device=args.bank_device
    if bank_device=="auto":
        bank_device="cpu" if bank["X"].numel()*bank["X"].element_size()>4e9 else device
    if bank_device=="cuda":
        bank_device=device
    X=bank["X"].to(bank_device)
    if X.device==torch.device(device):
        X=X.float()
    Xr=X.reshape(-1,*shape)
    vae=load_bank_vae(bank,device)
    print(f"bank {args.bank}: X {tuple(X.shape)} {X.dtype} on {X.device}, shape {shape}")

    # conditional networks run with the empty caption, encoded by the bank's CLIP
    clip_model,clip_processor=load_clip(bank["clip_model"],device)
    clip_fn=lambda texts:encode_texts_clip_tokens(clip_model,clip_processor,texts,device)
    nets={}
    for spec in [args.model]+args.other_models:
        label="target" if spec==args.model else spec
        nets[label]=TrainedNet(spec,device,clip_fn)
    t_min=nets["target"].t_min
    print(f"target {args.model}; all methods run from t={t_min:.4f}")

    methods=[]
    for m in args.methods:
        if m=="cfdm" and X.device!=torch.device(device):
            print("skipping cfdm: it needs the bank on the GPU")
        elif m in ["wiener","locality"] and X.shape[1]>20000:
            print(f"skipping {m}: d={X.shape[1]} is too large for d x d matrices")
        else:
            methods.append(m)

    generator=torch.Generator(device=device).manual_seed(args.seed)
    sigma=args.sigma*median_patch_distance(X,shape,1,generator=torch.Generator(device=X.device).manual_seed(args.seed))
    spectrum=None
    if "wiener" in methods or "locality" in methods:
        Xf=X.to(device).float()
        mu=Xf.mean(0)
        spectrum=global_spectrum((Xf.T@Xf-Xf.shape[0]*torch.outer(mu,mu))/(Xf.shape[0]-1))
        del Xf
    ls_schedule=patch_schedule_from_fracs(args.ls_patch_fracs or CELEBA_PATCH_FRACS,shape[-1],t_min)

    def run(method:str,z0:torch.Tensor)->torch.Tensor:
        B=z0.shape[0]
        if method in nets:
            return nets[method].sample(z0,args.steps,t_min)
        if method=="ideal":
            return machine_sample("ideal",z0,Xr,None,args.steps,t_min=t_min)
        if method=="ls":
            return machine_sample("ls",z0,Xr,ls_schedule,args.steps,t_min=t_min,chunk=args.ls_chunk)
        if method=="cfdm":
            return sample(B,X,None,w=0,sigma=sigma,M=args.M,steps=args.steps,start_t=t_min,generator=generator,z0=z0.flatten(1)).reshape(z0.shape)
        # wiener / locality with the global covariance: shrinkage 1 leaves only the global part
        out=locality_sample_shrunk([(X.to(device).float(),B,None)],*spectrum,1.0,0.0,method,args.tau,args.steps,t_min,z0s=[z0.flatten(1)])[0]
        return out.reshape(z0.shape)

    names=list(nets)+methods
    outputs={m:[] for m in names}
    seconds={m:0.0 for m in names}
    noise_gen=torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for i in range(0,args.n,args.batch):
            z0=torch.randn((min(args.batch,args.n-i),*shape),generator=noise_gen).to(device)
            for m in names:
                start=time.time()
                outputs[m].append(run(m,z0).float().cpu())
                if device=="cuda":
                    torch.cuda.synchronize()
                seconds[m]+=time.time()-start
            print(f"\t {min(i+args.batch,args.n)}/{args.n} samples, elapsed {sum(seconds.values()):.0f}s")
    outputs={m:torch.cat(v) for m,v in outputs.items()}

    target=outputs["target"]
    results={}
    for m in names:
        if m=="target":
            continue
        r=r2(outputs[m],target)
        mse=(outputs[m]-target).flatten(1).pow(2).mean(-1)
        results[m]={"r2_mean":float(r.mean()),"r2_std":float(r.std()),"mse_mean":float(mse.mean()),"mse_std":float(mse.std()),
                    "seconds_per_sample":seconds[m]/args.n}
    lines=[f"target: {args.model}","","| method | r2 vs target | MSE vs target | seconds/sample |","|---|---|---|---|"]
    for m,v in results.items():
        lines.append(f"| {m} | {v['r2_mean']:.3f} ± {v['r2_std']:.3f} | {v['mse_mean']:.4f} ± {v['mse_std']:.4f} | {v['seconds_per_sample']:.2f} |")
    print("\n".join(lines))

    os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,"agreement.json"),"w") as f:
        json.dump({"args":vars(args),"t_min":t_min,"results":results},f,indent=4)
    with open(os.path.join(args.output_dir,"agreement.md"),"w") as f:
        f.write("\n".join(lines)+"\n")
    # one row per method (target first), same noise in each column
    rows=[concat_images_horizontally(to_pil(bank_to_pixels(bank,outputs[m][:args.n_show].flatten(1),vae))) for m in names]
    concat_images_vertically(rows).save(os.path.join(args.output_dir,"agreement.png"))
    print(f"saved agreement.json, agreement.md and agreement.png (rows: {', '.join(names)}) to {args.output_dir}")


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
