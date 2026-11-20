'''
sample text-conditioned closed-form diffusion from a bank built by embed.py
usage: sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompt_file prompts/faces.txt --w 0 1 4
   or: sbatch runpygpu_chip.sh sample.py --bank banks/celebahq_pix64.pt --method pyramid --heatmap_lambda 1 --w 0 1 4
each output row is: [top CLIP-retrieved training image | samples...] for one (prompt, w); --unsharp_amount adds the sharpened samples below
--method cfdm: smoothed CFDM over the prompt's top-k CLIP matches
--method pyramid: that CFDM at the coarsest pyramid level, then closed-form patch levels up to full resolution (pyramid.py),
    with patches optionally weighted (--heatmap_lambda) and filtered (--heatmap_keep, --heatmap_random) by CLIP text heatmaps
--method net: a trained diffusion model
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
from cfdm import text_log_prior,top_k_subset,sample,median_distance
from pyramid import pyramid_sizes,downsample_flat,build_pyramid_banks,sample_pyramid,resize
from heatmap import token_heatmaps
from bank import load_clip,encode_texts_clip,encode_images_clip,load_bank_vae,bank_to_pixels,to_pil,unsharp_mask
from metrics import ReferenceMetrics,DinoCopy,LpipsDiversity,prompt_accuracy

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celeb_vae_256.pt")
parser.add_argument("--prompt_file",type=str,default="prompts/faces.txt",help="one prompt per line; blank lines are skipped")
parser.add_argument("--prompts",type=str,nargs="+",default=None,help="overrides --prompt_file")
parser.add_argument("--n_per_prompt",type=int,default=8)
parser.add_argument("--w",type=float,nargs="+",default=[0.0,1.0,4.0],help="text guidance weights to sweep; 0=unconditional, 1=exact conditional")
parser.add_argument("--method",type=str,default="cfdm",choices=["cfdm","pyramid","net"],help="cfdm: smoothed CFDM at full resolution; pyramid: CFDM at --coarse_size, then closed-form patch levels; net: a trained diffusion model (--model)")
parser.add_argument("--sigma",type=float,default=0.25,help="CFDM smoothing, as a fraction of the median pairwise distance between bank points (at the coarsest level for --method pyramid); 0 reproduces training points exactly")
parser.add_argument("--M",type=int,default=4,help="noise samples per smoothed score evaluation")
parser.add_argument("--steps",type=int,default=50,help="CFDM Euler steps (and the net's sampling steps)")
parser.add_argument("--start_t",type=float,default=0.2,help="CFDM starts from the exact mixture at this t (0 = pure noise)")
parser.add_argument("--top_k",type=int,default=1000,help="restrict the CFDM to the top-k CLIP matches per prompt (<=0 for all); w=0 always uses the full bank")
parser.add_argument("--coarse_size",type=int,default=16,help="pyramid: short side of the coarsest level, where the CFDM runs; levels halve from full resolution down to it")
parser.add_argument("--patch_size",type=int,default=7,help="pyramid: side of the square patches at every patch level")
parser.add_argument("--patch_images",type=int,default=50,help="pyramid: images supplying patches, the best CLIP matches (w=0: uniformly random ones); cost grows linearly with it")
parser.add_argument("--pyramid_steps",type=int,default=10,help="pyramid: denoising steps per patch level")
parser.add_argument("--fold_rho",type=float,default=0.2,help="pyramid: width of the Gaussian window overlapping patch estimates are averaged with")
parser.add_argument("--eta",type=float,default=0.0,help="pyramid: patch-level stochasticity in [0,1] (0 = deterministic DDIM)")
parser.add_argument("--heatmap_lambda",type=float,default=0.0,help="pyramid: logit bonus per standard deviation of a patch's CLIP text-heatmap score (0 = off)")
parser.add_argument("--heatmap_keep",type=float,default=1.0,help="pyramid: keep this fraction of patches, those with the highest heatmap scores (1 = all)")
parser.add_argument("--heatmap_random",type=float,default=0.0,help="pyramid: with --heatmap_keep < 1, also keep this random fraction of the other patches")
parser.add_argument("--heatmap_layers",type=int,default=4,help="pyramid: last attention layers whose direct effects make the heatmap")
parser.add_argument("--heatmap_baseline",type=str,nargs="+",default=["a photo.","an image.","a picture."],help="pyramid: generic texts whose heatmaps, with the other prompts', are averaged and subtracted")
parser.add_argument("--model",type=str,default="",help="--method net: 'pretrained:<repo>' (e.g. pretrained:google/ddpm-cifar10-32, as-is), a finetune.py path prefix (cross-attention models use the prompt with w as the CFG scale), or an --unconditional train.py prefix")
parser.add_argument("--bank_device",type=str,default="auto",choices=["auto","cpu","cuda"],help="where the bank lives; auto keeps banks over 4GB (e.g. 256px pixels) in CPU memory and moves only what's needed")
parser.add_argument("--fid_samples",type=int,default=0,help="if > 0, generate this many extra samples per w (split across prompts) for FID/KID")
parser.add_argument("--fid_ref_size",type=int,default=2000,help="reference images for FID/KID: held-out (X_test) and a random training subset")
parser.add_argument("--kid_subset",type=int,default=100,help="KID subset size (capped at --fid_samples)")
parser.add_argument("--fid_batch",type=int,default=64,help="samples generated per call during the FID pass")
parser.add_argument("--pr_k",type=int,default=3,help="k for the k-NN precision/recall balls")
parser.add_argument("--dino_model",type=str,default="facebook/dinov2-small",help="copy detection in this feature space during the FID pass ('' disables); training features are cached next to the bank")
parser.add_argument("--dino_copy_threshold",type=float,default=0.95,help="DINO cosine similarity to the nearest training image above which a sample counts as a copy")
parser.add_argument("--lpips_net",type=str,default="alex",help="LPIPS backbone for perceptual diversity: alex (the standard), vgg or squeeze ('' disables)")
parser.add_argument("--copy_threshold",type=float,default=1e-2,help="relative distance below which a sample counts as a training copy")
parser.add_argument("--eval_clip_model",type=str,default="",help="CLIP used for the clip_score metric; defaults to the conditioning CLIP (biased)")
parser.add_argument("--unsharp_amount",type=float,default=0.0,help="if > 0, also show the samples after an unsharp mask of this strength (e.g. 1.0) under the originals in the grids and wandb; metrics use the originals")
parser.add_argument("--unsharp_radius",type=float,default=1.0,help="unsharp mask Gaussian sigma in pixels (e.g. 1 at 64px, 2-3 at 256px)")
parser.add_argument("--unsharp_threshold",type=float,default=0.0,help="unsharp mask: only sharpen where the detail exceeds this (in [0,1] intensity units)")
parser.add_argument("--seed",type=int,default=123)
parser.add_argument("--output_dir",type=str,default="outputs")
parser.add_argument("--project_name",type=str,default="notrain")
parser.add_argument("--no_wandb",action="store_true")


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    bank=torch.load(args.bank,weights_only=False)
    bank_device=args.bank_device
    if bank_device=="auto":
        bank_device="cpu" if bank["X"].numel()*bank["X"].element_size()>4e9 else device
    if bank_device=="cuda":
        bank_device=device
    X=bank["X"].to(bank_device)
    T=bank["T"].to(device)
    if X.device!=torch.device(device) and args.method=="cfdm":
        raise ValueError("whole-bank cfdm needs the bank on the GPU; use --method pyramid or --bank_device cuda")
    shape=bank["shape"]
    print(f"bank {args.bank}: X {tuple(X.shape)} T {tuple(T.shape)} space {bank['space']} shape {shape}")

    if args.prompts is not None:
        prompts=args.prompts
    else:
        with open(args.prompt_file,"r") as file:
            prompts=[line.strip() for line in file if line.strip()]
    print("prompts",prompts)

    if not args.no_wandb:
        wandb.init(project=args.project_name,config=vars(args))

    clip_model,clip_processor=load_clip(bank["clip_model"],device)
    vae=load_bank_vae(bank,device)
    text_embeds=encode_texts_clip(clip_model,clip_processor,prompts,device)
    log_priors=text_log_prior(text_embeds,T,bank["logit_scale"])
    if args.eval_clip_model:
        eval_model,eval_processor=load_clip(args.eval_clip_model,device)
        eval_text_embeds=encode_texts_clip(eval_model,eval_processor,prompts,device)
    else:
        eval_model,eval_processor,eval_text_embeds=clip_model,clip_processor,text_embeds

    if args.method=="net":
        assert args.model,"--method net needs --model"
        from pretrained import TrainedNet
        from bank import encode_texts_clip_tokens
        net=TrainedNet(args.model,device,lambda texts:encode_texts_clip_tokens(clip_model,clip_processor,texts,device))
        print(f"trained net {args.model}: {'text-conditional (w = CFG scale)' if net.conditional else 'unconditional (prompts and w ignored)'}, starting at t={net.t_min:.4f}")
    lpips_metric=LpipsDiversity(device,args.lpips_net) if args.lpips_net else None

    # typical distance between bank points: sets the smoothing and normalizes the copy metric
    generator=torch.Generator(device=device).manual_seed(args.seed)
    scale=median_distance(X,generator=generator)
    sigma=args.sigma*scale
    if args.method=="pyramid":
        # the whole bank at the coarsest level, once: the CFDM runs there, and its smoothing is set there
        sizes=pyramid_sizes(shape[1],shape[2],args.coarse_size)
        coarse_shape=(shape[0],*sizes[-1])
        X_coarse=downsample_flat(X,shape,sizes[-1],device)
        sigma=args.sigma*median_distance(X_coarse,generator=generator)
        print(f"pyramid levels {sizes} (CFDM at {sizes[-1]}, {args.patch_size}x{args.patch_size} patches above)")
        heatmap_on=args.heatmap_lambda!=0 or args.heatmap_keep<1
        if heatmap_on:
            # every prompt's heatmaps, minus their mean over all prompts and the generic baseline texts
            heat_texts=torch.cat([text_embeds,encode_texts_clip(clip_model,clip_processor,args.heatmap_baseline,device)])
        pyramid_banks={}
    print(f"median pairwise distance {scale:.4f}, sigma {sigma:.4f}")

    os.makedirs(args.output_dir,exist_ok=True)
    # best CLIP matches per prompt (for the retrieved image shown in each row); the subset itself is only built when used
    top_idx={p:torch.topk(log_priors[p],min(max(args.top_k,args.patch_images if args.method=="pyramid" else 1,1),X.shape[0])).indices for p in range(len(prompts))}

    def patch_banks(p:int,w:float):
        '''
        the pyramid's patch banks for prompt p at weight w, built once and reused (e.g. across FID batches)
        '''
        if (p,w) in pyramid_banks:
            return pyramid_banks[(p,w)]
        if w==0:
            # unconditional: random images, no text prior or heatmap (the CLIP matches would themselves be conditioning)
            idx=torch.randperm(X.shape[0],generator=torch.Generator().manual_seed(args.seed+p))[:args.patch_images]
            prior,heat=None,None
        else:
            idx=top_idx[p][:args.patch_images].cpu()
            prior=log_priors[p][idx.to(device)]
            heat=None
            if heatmap_on:
                images=to_pil(bank_to_pixels(bank,X[idx.to(X.device)],vae))
                maps=token_heatmaps(clip_model,clip_processor,images,heat_texts,device,args.heatmap_layers)
                heat=resize((maps[:,p]-maps.mean(1))[:,None],shape[1:])
        images=X[idx.to(X.device)].to(device).float().reshape(-1,*shape)
        # no heatmap (w=0) means no filtering either
        keep=args.heatmap_keep if heat is not None else 1.0
        banks=build_pyramid_banks(images,args.coarse_size,args.patch_size,prior,w,heat,args.heatmap_lambda,
                                  keep,args.heatmap_random,generator)
        info={"patches_per_level":[b["patches"].shape[0] for b in banks],
              # no patch levels when --coarse_size is the full resolution: the CFDM alone
              "patches_kept":sum(b["patches"].shape[0] for b in banks)/max(sum(b["n_total"] for b in banks),1)}
        print(f"prompt {p} w={w}: patch banks {info}")
        pyramid_banks[(p,w)]=(banks,info)
        return banks,info

    def generate(p:int,w:float,n:int):
        '''
        n samples for prompt p at weight w; returns (samples, prior_mass or None, extra metrics)
        '''
        if args.method=="net":
            z0=torch.randn((n,*shape),generator=generator,device=device)
            return net.sample(z0,args.steps,net.t_min,prompts[p],w).flatten(1),None,{}
        if w==0:
            # unconditional baseline: the top-k subset would itself be text conditioning
            sub,log_prior,prior_mass=None,None,1.0
        else:
            _,log_prior,sub=top_k_subset(T,log_priors[p],args.top_k)
            # fraction of the w-sharpened prior that the top-k subset keeps
            prior_mass=torch.softmax(w*log_priors[p],dim=-1)[sub].sum().item()
            if prior_mass<0.9:
                print(f"warning: top_k={args.top_k} keeps only {prior_mass:.3f} of the prior mass for '{prompts[p]}' at w={w}")
        if args.method=="cfdm":
            X_w=X if sub is None else X[sub]
            out=sample(n,X_w,log_prior,w=w,sigma=sigma,M=args.M,steps=args.steps,start_t=args.start_t,generator=generator)
            return out,prior_mass,{}
        banks,info=patch_banks(p,w)
        coarse=X_coarse if sub is None else X_coarse[sub]
        out=sample_pyramid(n,coarse,shape,banks,args.coarse_size,args.patch_size,log_prior,w=w,sigma=sigma,M=args.M,
                           cfdm_steps=args.steps,start_t=args.start_t,steps=args.pyramid_steps,rho=args.fold_rho,
                           eta=args.eta,generator=generator)
        return out,prior_mass,info

    def nearest_train(out:torch.Tensor)->torch.Tensor:
        # distance to the closest training point, relative to the typical pairwise distance; streams the bank
        # through the GPU in chunks so it also works when X lives on the CPU
        best=torch.full((out.shape[0],),float("inf"),device=out.device)
        for j in range(0,X.shape[0],4096):
            Xc=X[j:j+4096].to(out.device).float()
            best=torch.minimum(best,torch.cdist(out.float(),Xc).min(-1).values)
        return best/scale

    results=[]
    for p,prompt in enumerate(prompts):
        name=re.sub(r"[^A-Za-z0-9]+","_",prompt).strip("_")[:60]
        retrieved=to_pil(bank_to_pixels(bank,X[top_idx[p][:1].to(X.device)],vae))[0]
        rows=[]
        for w in args.w:
            start=time.time()
            out,prior_mass,info=generate(p,w,args.n_per_prompt)
            if device=="cuda":
                torch.cuda.synchronize()
            elapsed=time.time()-start
            nearest=nearest_train(out)
            pixels=bank_to_pixels(bank,out,vae)
            images=to_pil(pixels)
            image_embeds=encode_images_clip(eval_model,eval_processor,images,device)
            clip_scores=(image_embeds@eval_text_embeds[p]).cpu().numpy()
            pair=torch.cdist(out,out)
            metrics={
                "prompt":prompt,
                "w":w,
                "prior_mass":prior_mass,
                "clip_score":float(np.mean(clip_scores)),
                "copy_rate":float((nearest<args.copy_threshold).float().mean()),
                "nn_dist":float(nearest.mean()),
                # mean distance between different samples of this prompt (0 = all identical)
                "diversity":float(pair.sum()/max(out.shape[0]*(out.shape[0]-1),1)/scale),
                # perceptual version of the same, and whether an independent CLIP picks this prompt among all prompts
                "lpips_diversity":lpips_metric.mean_pairwise(pixels) if lpips_metric else None,
                "prompt_accuracy":prompt_accuracy(image_embeds,eval_text_embeds,torch.full((out.shape[0],),p)),
                # the first call includes building the pyramid's patch banks
                "seconds_per_sample":elapsed/args.n_per_prompt,
                **info
            }
            print(metrics)
            results.append(metrics)
            row=concat_images_horizontally([retrieved]+images)
            if args.unsharp_amount>0:
                # before (top) and after (bottom) sharpening
                sharpened=to_pil(unsharp_mask(pixels,args.unsharp_amount,args.unsharp_radius,args.unsharp_threshold))
                row=concat_images_vertically([row,concat_images_horizontally([retrieved]+sharpened)])
            rows.append(row)
            if not args.no_wandb:
                wandb.log({f"{name}/w={w}":wandb.Image(row),**{f"{name}/{k}":v for k,v in metrics.items() if k!="prompt" and not isinstance(v,list)}})
        grid=concat_images_vertically(rows)
        grid.save(os.path.join(args.output_dir,f"{p}_{name}.png"))

    if args.fid_samples>0:
        # FID/KID over fid_samples samples per w, split evenly across prompts, against held-out images
        # (embed.py --holdout) and against training images; "vs train" much lower than "vs test" means copying
        ref_gen=torch.Generator().manual_seed(args.seed)
        refs={"train":X[torch.randperm(X.shape[0],generator=ref_gen)[:args.fid_ref_size].to(X.device)].float()}
        if bank.get("X_test") is not None and bank["X_test"].shape[0]>0:
            refs["test"]=bank["X_test"][:args.fid_ref_size].float()
        else:
            print("warning: bank has no held-out images (embed.py --holdout), FID is only against training images")
        kid_subset=min(args.kid_subset,args.fid_samples)
        ref_metrics={k:ReferenceMetrics(bank_to_pixels(bank,v,vae).to(device),device,kid_subset,pr_k=args.pr_k) for k,v in refs.items()}
        dino=None
        if args.dino_model:
            dino=DinoCopy(args.dino_model,device)
            cache_path=args.bank+"."+args.dino_model.replace("/","_")+".pt"
            dino.set_train(lambda i,j:bank_to_pixels(bank,X[i:j],vae),X.shape[0],cache_path)
        per_prompt=[args.fid_samples//len(prompts)+(1 if p<args.fid_samples%len(prompts) else 0) for p in range(len(prompts))]
        for w in args.w:
            for m in ref_metrics.values():
                m.reset()
            nearest=[]
            all_pixels=[]
            prompt_ids=[]
            start=time.time()
            for p,n_p in enumerate(per_prompt):
                for i in range(0,n_p,args.fid_batch):
                    out,_,_=generate(p,w,min(args.fid_batch,n_p-i))
                    nearest.append(nearest_train(out))
                    pixels=bank_to_pixels(bank,out,vae).to(device)
                    for m in ref_metrics.values():
                        m.update(pixels)
                    all_pixels.append(pixels.cpu())
                    prompt_ids.append(torch.full((out.shape[0],),p))
            generate_seconds=time.time()-start
            nearest=torch.cat(nearest)
            all_pixels=torch.cat(all_pixels)
            prompt_ids=torch.cat(prompt_ids)
            image_embeds=torch.cat([encode_images_clip(eval_model,eval_processor,to_pil(all_pixels[i:i+64]),device)
                                    for i in range(0,all_pixels.shape[0],64)])
            metrics={"prompt":"__all__","w":w,"fid_samples":args.fid_samples,
                     "fid_copy_rate":float((nearest<args.copy_threshold).float().mean()),"fid_nn_dist":float(nearest.mean()),
                     "fid_seconds_per_sample":generate_seconds/args.fid_samples,
                     "fid_prompt_accuracy":prompt_accuracy(image_embeds,eval_text_embeds,prompt_ids)}
            if dino is not None:
                sims=dino.nearest_similarity(all_pixels)
                metrics["dino_nn_sim"]=float(sims.mean())
                metrics["dino_copy_rate"]=float((sims>=args.dino_copy_threshold).float().mean())
            if lpips_metric is not None:
                per=[lpips_metric.mean_pairwise(all_pixels[prompt_ids==p]) for p in range(len(prompts))]
                per=[v for v in per if v is not None]
                metrics["fid_lpips_diversity"]=float(np.mean(per)) if per else None
            for k,m in ref_metrics.items():
                for name_,v in m.compute().items():
                    metrics[f"{name_}_{k}"]=v
            print(metrics)
            results.append(metrics)
            if not args.no_wandb:
                wandb.log({f"fid/w={w}/{k}":v for k,v in metrics.items() if k not in ["prompt","w"]})

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
