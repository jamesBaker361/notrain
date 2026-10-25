'''
how well do closed-form machines (els.py) predict a trained text-conditioned network (train.py)?

1. run the network's Euler sampler for each (prompt, seed) and record every z_t with the network's x0_hat
2. feed the same z_t to each machine (ideal / ls / els, at each patch size) and record r^2 against the network
3. calibrate the patch size per step on the even seeds, report r^2 on the odd seeds
4. run each machine's own sampler from the same initial noise with the calibrated patch schedule and
   compare final samples (r^2 in latent space + decoded image grids)
prompts: --n_train_prompts captions drawn from the training set, plus any --prompts / --prompt_file (novel text)
usage: sbatch runpygpu_chip.sh compare.py --bank banks/celeb.pt --model weights/celeb_conv/ema --output_dir compare/celeb_conv
'''
import os
import csv
import json
import argparse
import time
import random
import numpy as np
import torch
from experiment_helpers.gpu_details import print_details
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
from bank import load_clip,encode_texts_clip,encode_texts_clip_tokens,load_bank_vae,bank_to_pixels,to_pil
from textunet import TextUNet
from els import denoise,r2,machine_sample,text_log_prior_from

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celeb.pt")
parser.add_argument("--model",type=str,default="weights/celeb_conv/ema",help="path prefix written by train.py (no .pt/.json)")
parser.add_argument("--n_train_prompts",type=int,default=4,help="training captions to use as prompts")
parser.add_argument("--prompts",type=str,nargs="+",default=None)
parser.add_argument("--prompt_file",type=str,default="")
parser.add_argument("--seeds",type=int,default=4,help="noise seeds per prompt; even seeds calibrate, odd seeds test")
parser.add_argument("--steps",type=int,default=50)
parser.add_argument("--eval_every",type=int,default=2,help="compare denoisers at every k-th step")
parser.add_argument("--guidance",type=float,default=1.0,help="network CFG scale (1 = plain conditional)")
parser.add_argument("--machines",type=str,nargs="+",default=["ideal","ls","els"],choices=["ideal","ls","els"])
parser.add_argument("--patch_sizes",type=int,nargs="+",default=[1,3,5,7,9,13,17,25],help="patch sizes for ls")
parser.add_argument("--els_patch_sizes",type=int,nargs="+",default=[1,3,5,7,9],help="patch sizes for els, whose cost grows with P^2 * (els_images * H * W)")
parser.add_argument("--prior",type=str,default="caption",choices=["caption","image","none"],help="text prior over training images: CLIP sim to their captions, to their images, or none")
parser.add_argument("--beta",type=float,default=-1,help="prior inverse temperature; -1 uses the CLIP logit scale")
parser.add_argument("--w",type=float,default=1.0,help="exponent on the prior (the closed-form analog of guidance)")
parser.add_argument("--els_images",type=int,default=256,help="candidate images for els: top by prior, or random without a prior")
parser.add_argument("--t_min",type=float,default=1e-3)
parser.add_argument("--output_dir",type=str,default="compare")
parser.add_argument("--seed",type=int,default=123)


def network_trajectory(model,z0,cond,uncond,guidance,steps,t_min):
    '''
    Euler on v = (x0_hat - z)/(1-t); returns [(t, z_t, x0_hat)] and the final sample
    cond/uncond: (text, mask) tuples batched to z0
    '''
    ts=torch.linspace(t_min,1.0,steps+1).tolist()
    z=z0
    record=[]
    for t,t_next in zip(ts[:-1],ts[1:]):
        tt=torch.full((z.shape[0],),t,device=z.device)
        x0=model.x0(z,tt,*cond)
        if guidance!=1.0:
            x0_u=model.x0(z,tt,*uncond)
            x0=x0_u+guidance*(x0-x0_u)
        record.append((t,z,x0))
        z=z+(t_next-t)*(x0-z)/(1-t)
    return record,z


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    bank=torch.load(args.bank,weights_only=False)
    model,meta=TextUNet.load(args.model,map_location=device)
    model=model.to(device).eval()
    n=meta["n_train"]
    shape=bank["shape"]
    X=bank["X"][:n].reshape(n,*shape).to(device)
    T=bank["T"][:n].to(device)
    C=bank["C"][:n].to(device)
    print(f"model {args.model} ({'attention' if meta['attention'] else 'conv-only'}, step {meta.get('step')}), {n} training images")

    prompts=[(c,"train") for c in random.sample(bank["captions"][:n],min(args.n_train_prompts,n))]
    novel=list(args.prompts or [])
    if args.prompt_file:
        with open(args.prompt_file) as f:
            novel+=[line.strip() for line in f if line.strip()]
    prompts+=[(p,"novel") for p in novel]
    print("prompts",prompts)

    clip_model,clip_processor=load_clip(bank["clip_model"],device)
    vae=load_bank_vae(bank,device)
    texts=[p for p,_ in prompts]
    pooled=encode_texts_clip(clip_model,clip_processor,texts,device)
    beta=bank["logit_scale"] if args.beta<0 else args.beta
    keys={"caption":C,"image":T}.get(args.prior)
    log_priors=None if keys is None else text_log_prior_from(pooled,keys,beta)

    def net_text(i,B):
        if meta["attention"]:
            states,mask=encode_texts_clip_tokens(clip_model,clip_processor,[texts[i]] if i is not None else [""],device)
            return states.expand(B,-1,-1),mask.expand(B,-1)
        emb=pooled[i] if i is not None else encode_texts_clip(clip_model,clip_processor,[""],device)[0]
        return emb[None].expand(B,-1),None

    os.makedirs(args.output_dir,exist_ok=True)
    rows=[]
    finals={}
    configs=[(m,None) for m in args.machines if m=="ideal"]+[(m,P) for m in args.machines if m!="ideal" for P in (args.els_patch_sizes if m=="els" else args.patch_sizes)]
    start=time.time()
    with torch.no_grad():
        for i,(prompt,kind) in enumerate(prompts):
            log_prior=None if log_priors is None else log_priors[i]
            # els only sees a subset of images: the top ones by prior, or a fixed random subset
            if log_prior is not None:
                els_idx=torch.topk(log_prior,min(args.els_images,n)).indices
                els_prior=torch.log_softmax(log_prior[els_idx],-1)
            else:
                els_idx=torch.randperm(n,generator=torch.Generator().manual_seed(args.seed))[:args.els_images].to(device)
                els_prior=None
            for seed in range(args.seeds):
                z0=torch.randn((1,*shape),generator=torch.Generator().manual_seed(1000*i+seed)).to(device)
                record,final=network_trajectory(model,z0,net_text(i,1),net_text(None,1),args.guidance,args.steps,args.t_min)
                finals[(i,seed)]={"z0":z0,"network":final}
                for step,(t,z,x0_net) in enumerate(record):
                    if step%args.eval_every!=0:
                        continue
                    for machine,P in configs:
                        if machine=="els":
                            x0=denoise("els",z,X[els_idx],t,P,els_prior,args.w)
                        else:
                            x0=denoise(machine,z,X,t,P,log_prior,args.w)
                        rows.append({"prompt":i,"kind":kind,"seed":seed,"step":step,"t":t,"machine":machine,
                                     "P":P if P is not None else 0,"r2":float(r2(x0,x0_net)[0])})
                print(f"\t prompt {i} seed {seed} done, {len(rows)} rows, elapsed {time.time()-start:.0f}s")

    with open(os.path.join(args.output_dir,"rows.csv"),"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # calibrate P per (machine, step) on even seeds, evaluate on odd seeds
    summary={}
    schedules={}
    for machine in args.machines:
        mrows=[r for r in rows if r["machine"]==machine]
        steps=sorted({r["step"] for r in mrows})
        best_P,test_r2,ts=[],[],[]
        for step in steps:
            srows=[r for r in mrows if r["step"]==step]
            calib={}
            for r in srows:
                if r["seed"]%2==0:
                    calib.setdefault(r["P"],[]).append(r["r2"])
            P=max(calib,key=lambda p:np.mean(calib[p])) if calib else srows[0]["P"]
            test=[r["r2"] for r in srows if r["P"]==P and (r["seed"]%2==1 or args.seeds==1)]
            best_P.append(P)
            test_r2.append(float(np.mean(test)))
            ts.append(srows[0]["t"])
        schedules[machine]=(ts,best_P)
        summary[machine]={"mean_test_r2":float(np.mean(test_r2)),"t":ts,"best_P":best_P,"test_r2":test_r2}
        print(f"{machine}: mean held-out r2 {np.mean(test_r2):.3f}; best P by t {list(zip([round(t,2) for t in ts],best_P))}")

    # final samples: each machine's own sampler from the same noise, with its calibrated patch schedule
    grid_rows=[]
    final_r2={m:[] for m in args.machines}
    with torch.no_grad():
        for (i,seed),f in finals.items():
            log_prior=None if log_priors is None else log_priors[i]
            outs=[f["network"]]
            for machine in args.machines:
                ts,Ps=schedules[machine]
                schedule=lambda t,ts=ts,Ps=Ps:Ps[int(np.argmin([abs(t-s) for s in ts]))]
                if machine=="els":
                    if log_prior is not None:
                        idx=torch.topk(log_prior,min(args.els_images,n)).indices
                        out=machine_sample("els",f["z0"],X[idx],schedule,args.steps,torch.log_softmax(log_prior[idx],-1),args.w,args.t_min)
                    else:
                        idx=torch.randperm(n,generator=torch.Generator().manual_seed(args.seed))[:args.els_images].to(device)
                        out=machine_sample("els",f["z0"],X[idx],schedule,args.steps,None,args.w,args.t_min)
                else:
                    out=machine_sample(machine,f["z0"],X,schedule,args.steps,log_prior,args.w,args.t_min)
                final_r2[machine].append(float(r2(out,f["network"])[0]))
                outs.append(out)
            images=to_pil(bank_to_pixels(bank,torch.cat(outs).flatten(1),vae))
            grid_rows.append(concat_images_horizontally(images))
    for machine in args.machines:
        summary[machine]["final_sample_r2"]=float(np.mean(final_r2[machine]))
        print(f"{machine}: final-sample r2 vs network {np.mean(final_r2[machine]):.3f}")
    concat_images_vertically(grid_rows).save(os.path.join(args.output_dir,"finals.png"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        for machine in args.machines:
            ax.plot(summary[machine]["t"],summary[machine]["test_r2"],label=machine)
        ax.set_xlabel("t (0 = noise, 1 = data)")
        ax.set_ylabel("held-out r$^2$ vs network x$_0$")
        ax.set_ylim(min(0,ax.get_ylim()[0]),1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir,"r2_vs_t.png"),dpi=150)
    except ImportError:
        print("matplotlib missing, skipping plot")

    with open(os.path.join(args.output_dir,"summary.json"),"w") as f:
        json.dump({"args":vars(args),"model_meta":meta,"prompts":prompts,"machines":summary},f,indent=4)
    print(f"saved rows.csv, summary.json, r2_vs_t.png and finals.png (columns: network, {', '.join(args.machines)}) to {args.output_dir}")


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
