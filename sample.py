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
from bank import load_clip,encode_texts_clip,encode_images_clip,load_bank_vae,bank_to_pixels,to_pil
from tree import tree_sample,node_size,sample_node,subset_tree
from locality import global_spectrum
from metrics import ReferenceMetrics,DinoCopy,LpipsDiversity,prompt_accuracy

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
parser.add_argument("--method",type=str,default="cfdm",choices=["cfdm","tree","topk","net"],help="tree: descend a CLIP tree built by tree.py, then sample from the node; topk: no tree, sample from the prompt's --top_k CLIP matches with --node_sampler, each image weighted by softmax(w * log p(image | prompt)) (w=0: the K images unweighted); net: sample a trained diffusion model (--model)")
parser.add_argument("--model",type=str,default="",help="--method net: 'pretrained:<repo>' (e.g. pretrained:google/ddpm-cifar10-32, as-is), a finetune.py path prefix (cross-attention models use the prompt with w as the CFG scale), or an --unconditional train.py prefix")
parser.add_argument("--bank_device",type=str,default="auto",choices=["auto","cpu","cuda"],help="where the bank lives; auto keeps banks over 4GB (e.g. 256px pixels) in CPU memory and moves only what's needed")
parser.add_argument("--tree",type=str,default="",help="tree built by tree.py for this bank")
parser.add_argument("--tree_depth",type=int,default=5,help="stop the descent at this depth (L)")
parser.add_argument("--c_search",type=int,default=20,help="stop the descent at the first node with this many images or fewer")
parser.add_argument("--node_sampler",type=str,default="lowrank",choices=["diagonal","lowrank","cfdm","locality","wiener","ls","curve"],help="how to sample from the stopping node; cfdm uses --sigma, --M, --steps, --start_t; locality/wiener (Lukoianov et al.) use --steps, --tau, --wiener_shrinkage, --wiener_ridge; ls (Kamb & Ganguli) uses --steps, --ls_patch_fracs; curve (a spline through the node's images in TSP order) uses --curve_*, --tsp, --tsp_kicks, --steps")
parser.add_argument("--curve_mode",type=str,default="random",choices=["random","flow"],help="curve: pick a random point on the curve (text-weighted), or run the ideal-denoiser flow toward the curve")
parser.add_argument("--curve_max_points",type=int,default=2000,help="curve: nodes with more images fit the curve through a random subset of this many")
parser.add_argument("--curve_knots",type=int,default=0,help="curve: B-spline basis functions; 0 = one per image (the curve passes through every image), fewer = smoother curves that average neighboring images")
parser.add_argument("--curve_degree",type=int,default=1,help="curve: spline degree; 1 = polyline through the images (crossfades between path neighbors), 3 = cubic (smoother, but overshoots between dissimilar images)")
parser.add_argument("--curve_smooth",type=float,default=0.0,help="curve: second-difference penalty relative to the fit term (0 = least squares)")
parser.add_argument("--curve_grid",type=int,default=0,help="curve: points along the curve that samples are drawn from; 0 = 16 per image within a 2GB budget")
parser.add_argument("--tsp",type=str,default="auto",choices=["auto","2opt","lkh"],help="curve: image order by chained 2-opt, or LKH (pip install elkai); auto uses LKH if installed")
parser.add_argument("--tsp_kicks",type=int,default=30,help="curve: double-bridge kicks for chained 2-opt")
parser.add_argument("--ls_patch_fracs",type=float,nargs="+",default=None,help="ls: patch sizes from noise to data as fractions of the image side; default is the CelebA-64 schedule from Lukoianov et al.")
parser.add_argument("--tau",type=float,default=0.02,help="locality: Wiener-matrix binarization threshold, relative to each row's max")
parser.add_argument("--wiener_shrinkage",type=float,default=0.0,help="locality/wiener: blend the node covariance toward the full-bank covariance (0 = node only, 1 = global)")
parser.add_argument("--wiener_ridge",type=float,default=0.01,help="locality/wiener: add this fraction of the mean pixel variance to the covariance diagonal so small (rank-deficient) nodes still get local masks at low noise")
parser.add_argument("--wiener_rank",type=int,default=0,help="wiener: if > 0, use rank-k node (and global) covariances with the left-out variance kept as an isotropic term; no d x d matrices, so it runs at any d (e.g. 256px pixels)")
parser.add_argument("--wiener_exact_eig",action="store_true",help="with --wiener_shrinkage, eigendecompose each node's blended covariance instead of the (equivalent, faster) Woodbury update of the global one")
parser.add_argument("--node_text_prior",action="store_true",help="locality: also weight node images by w * log p(i|c) inside the softmax")
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
        raise ValueError("whole-bank cfdm needs the bank on the GPU; use --method tree or --bank_device cuda")
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
    vae=load_bank_vae(bank,device)
    text_embeds=encode_texts_clip(clip_model,clip_processor,prompts,device)
    log_priors=text_log_prior(text_embeds,T,bank["logit_scale"])
    if args.eval_clip_model:
        eval_model,eval_processor=load_clip(args.eval_clip_model,device)
        eval_text_embeds=encode_texts_clip(eval_model,eval_processor,prompts,device)
    else:
        eval_model,eval_processor,eval_text_embeds=clip_model,clip_processor,text_embeds

    if args.method=="topk":
        assert args.top_k>0,"--method topk needs --top_k > 0"
    if args.method in ["tree","topk"]:
        if args.method=="tree":
            assert args.tree,"--method tree needs --tree"
            tree=torch.load(args.tree,weights_only=False)
            assert tree["n"]==X.shape[0],f"tree was built for {tree['n']} images but the bank has {X.shape[0]}"
        eig_cache={}
        # topk: statistics depend on the prompt and w, so each (prompt, w) gets its own cache
        topk_caches={}
        S_global=None
        spectrum=None
        global_lowrank=None
        lowrank_wiener=args.node_sampler=="wiener" and args.wiener_rank>0
        if lowrank_wiener and args.wiener_shrinkage>0:
            from locality import low_rank_cov
            # top components of the whole bank, streamed from wherever the bank lives
            global_lowrank=low_rank_cov(X,args.wiener_rank,device)
            print(f"global low-rank covariance: {global_lowrank[2].shape[1]} components, residual variance {global_lowrank[3]:.4g}")
        if args.node_sampler in ["locality","wiener"] and args.wiener_shrinkage>0 and not lowrank_wiener:
            # global covariance without materializing a centered copy of the bank
            assert X.shape[1]<=20000,f"locality/wiener need d x d matrices; d={X.shape[1]} is too large (use pixel <=64px or the vq latents)"
            Xf=X.to(device).float()
            mu=Xf.mean(0)
            S_global=(Xf.T@Xf-Xf.shape[0]*torch.outer(mu,mu))/(Xf.shape[0]-1)
            del Xf
            if not args.wiener_exact_eig and args.method=="tree":
                # Woodbury path: one eigendecomposition for the run instead of one per node
                spectrum=global_spectrum(S_global)
                S_global=None

    if args.method=="net":
        assert args.model,"--method net needs --model"
        from pretrained import TrainedNet
        from bank import encode_texts_clip_tokens
        net=TrainedNet(args.model,device,lambda texts:encode_texts_clip_tokens(clip_model,clip_processor,texts,device))
        print(f"trained net {args.model}: {'text-conditional (w = CFG scale)' if net.conditional else 'unconditional (prompts and w ignored)'}, starting at t={net.t_min:.4f}")
    lpips_metric=LpipsDiversity(device,args.lpips_net) if args.lpips_net else None

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
    # best CLIP matches per prompt (for the retrieved image shown in each row); the subset itself is only built when used
    top_idx={p:torch.topk(log_priors[p],min(max(args.top_k,1),X.shape[0])).indices for p in range(len(prompts))}

    def generate(p:int,w:float,n:int):
        '''
        n samples for prompt p at weight w; returns (samples, tree nodes or None, prior_mass or None)
        '''
        if args.method=="net":
            z0=torch.randn((n,*bank["shape"]),generator=generator,device=device)
            return net.sample(z0,args.steps,net.t_min,prompts[p],w).flatten(1),None,None
        if args.method in ["tree","topk"]:
            cfdm_kwargs={"log_prior":log_priors[p],"w":w,"sigma":sigma,"M":args.M,"steps":args.steps,"start_t":args.start_t}
            locality_kwargs={"tau":args.tau,"steps":args.steps,"shrinkage":args.wiener_shrinkage,"ridge":args.wiener_ridge,
                             "S_global":S_global,"spectrum":spectrum,"cache":eig_cache,
                             "shape":bank["shape"],"ls_patch_fracs":args.ls_patch_fracs,
                             "wiener_rank":args.wiener_rank,"global_lowrank":global_lowrank,
                             "log_prior":log_priors[p] if args.node_text_prior else None,"w":w,
                             "curve":{"max_points":args.curve_max_points,"n_knots":args.curve_knots,"degree":args.curve_degree,"smooth":args.curve_smooth,
                                      "grid":args.curve_grid,"mode":args.curve_mode,"tsp":args.tsp,"kicks":args.tsp_kicks,
                                      "steps":args.steps,"log_prior":log_priors[p],"w":w}}
        if args.method=="tree":
            # w sets the descent sharpness (0 = uniform random walk), so no top-k subset is used
            out,nodes=tree_sample(tree,X,text_embeds[p],n,w,bank["logit_scale"],args.tree_depth,
                                  args.c_search,args.node_sampler,generator,cfdm_kwargs=cfdm_kwargs,
                                  locality_kwargs=locality_kwargs,device=device)
            return out,nodes,None
        if args.method=="topk":
            # the prompt's top-K CLIP matches as one node; wiener/locality statistics weighted by softmax(w * log prior)
            # (cfdm and curve weight by the same prior themselves); w=0 keeps the K images unweighted
            idx=top_idx[p]
            weights=torch.softmax(w*log_priors[p][idx].float(),dim=-1)
            locality_kwargs["weights"]=weights
            locality_kwargs["cache"]=topk_caches.setdefault((p,w),{})
            out=sample_node(subset_tree(idx),X,0,n,args.node_sampler,generator,cfdm_kwargs,locality_kwargs,device)
            prior_mass=torch.softmax(w*log_priors[p],dim=-1)[idx].sum().item()
            return out,None,prior_mass
        X_sub,log_prior,idx=top_k_subset(X,log_priors[p],args.top_k)
        if w==0:
            # unconditional baseline: the top-k subset would itself be text conditioning
            X_w,log_prior_w,prior_mass=X,None,1.0
        else:
            # fraction of the w-sharpened prior that the top-k subset keeps
            X_w,log_prior_w=X_sub,log_prior
            prior_mass=torch.softmax(w*log_priors[p],dim=-1)[idx].sum().item()
            if prior_mass<0.9:
                print(f"warning: top_k={args.top_k} keeps only {prior_mass:.3f} of the prior mass for '{prompts[p]}' at w={w}")
        if args.scales==[1]:
            out=sample(n,X_w,log_prior_w,w=w,sigma=sigma,M=args.M,steps=args.steps,start_t=args.start_t,generator=generator)
        else:
            out=sample_multiscale(n,X_w,bank["shape"],args.scales,log_prior_w,w=w,sigmas=sigmas,M=args.M,
                                  steps=args.steps,start_t=args.start_t,schedule=args.schedule,
                                  constant_weights=args.scale_weights,positions=args.patch_positions,generator=generator)
        return out,None,prior_mass

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
            out,nodes,prior_mass=generate(p,w,args.n_per_prompt)
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
