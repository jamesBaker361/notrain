'''
exact data attribution for the closed-form samplers, checked with exact counterfactuals
every retrieval, cfdm or pyramid sample is an exact convex combination of training data (samplers.py), so its
attribution is read off, not estimated: per bank image (content: share of the output pixels; layout: the CFDM
weights behind the arrangement) and, for the pyramid, per output pixel down to the source pixels. And removing
images from the bank *is* retraining without them, so attribution rankings can be scored against ground truth:
for each sample, delete the top-k images under a ranking, regenerate with the same noise, and measure how much the
sample changes (more = the ranking found the images the sample depends on). Rankings:
    attr / attr_content / attr_layout   ours (content + layout averaged, or either alone)
    dino / clip / pixel                 similarity of the sample to every training image (DINOv2, the bank's
                                        CLIP, distance in bank space): the usual post-hoc attribution
    random                              k random images among the prompt's top-k CLIP matches
"none" regenerates on the full bank: its change should be 0 (the counterfactuals share the original's noise)
writes attribution.json (every record), summary.md (mean and standard error per ranking and k, and how often
ours beats each baseline on the same sample) and, for the first --n_show samples per prompt, figure rows:
    sample | top-source mosaic (each pixel from its top source pixel) | provenance (each pixel coloured by its
    top source image among the 5 largest) | top-5 attributed images | sample without them | without the top-5 DINO
usage: sbatch runpygpu_chip.sh attribution.py --bank banks/celebahq_pix64.pt --method pyramid --prompt_file prompts/celeb.txt --output_dir outputs/attribution_celebahq_pix64
'''
import os
import json
import time
import argparse
import numpy as np
import torch
from experiment_helpers.gpu_details import print_details
from experiment_helpers.image_helpers import concat_images_horizontally,concat_images_vertically
from cfdm import text_log_prior
from samplers import ClosedForm,add_sampler_args
from bank import load_clip,encode_texts_clip,encode_images_clip,load_bank_vae,bank_to_pixels,to_pil,row_chunk
from metrics import DinoCopy,LpipsDiversity

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celebahq_pix64.pt")
parser.add_argument("--prompt_file",type=str,default="prompts/celeb.txt")
parser.add_argument("--prompts",type=str,nargs="+",default=None,help="overrides --prompt_file")
parser.add_argument("--method",type=str,default="pyramid",choices=["retrieval","cfdm","pyramid"])
parser.add_argument("--w",type=float,default=1.0,help="text guidance weight (> 0: the prompt's top matches)")
parser.add_argument("--n_per_prompt",type=int,default=8)
parser.add_argument("--ks",type=int,nargs="+",default=[1,5,20],help="images deleted per counterfactual")
parser.add_argument("--rankings",type=str,nargs="+",default=["attr","attr_content","attr_layout","dino","clip","pixel","random"])
parser.add_argument("--dino_model",type=str,default="facebook/dinov2-small")
parser.add_argument("--lpips_net",type=str,default="alex")
parser.add_argument("--n_show",type=int,default=2,help="samples per prompt shown in the figure")
parser.add_argument("--output_dir",type=str,default="outputs/attribution")
add_sampler_args(parser)

# provenance colours for the top-5 source images; everything else grey
PALETTE=torch.tensor([[0.90,0.10,0.10],[0.10,0.45,0.90],[0.10,0.70,0.20],[0.95,0.65,0.05],[0.60,0.20,0.80]])


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    assert args.w>0,"attribution runs on the prompt's top matches (--w > 0)"
    bank=torch.load(args.bank,weights_only=False)
    bank_device=args.bank_device
    if bank_device=="auto":
        bank_device="cpu" if bank["X"].numel()*bank["X"].element_size()>4e9 else device
    if bank_device=="cuda":
        bank_device=device
    X=bank["X"].to(bank_device)
    T=bank["T"].to(device)
    shape=bank["shape"]
    if args.prompts is not None:
        prompts=args.prompts
    else:
        with open(args.prompt_file) as f:
            prompts=[line.strip() for line in f if line.strip()]
    clip_model,clip_processor=load_clip(bank["clip_model"],device)
    vae=load_bank_vae(bank,device)
    text_embeds=encode_texts_clip(clip_model,clip_processor,prompts,device)
    log_priors=text_log_prior(text_embeds,T,bank["logit_scale"])
    sampler=ClosedForm(args,bank,X,T,log_priors,text_embeds,clip_model,clip_processor,device)
    lpips=LpipsDiversity(device,args.lpips_net).model if args.lpips_net else None
    dino=None
    if args.dino_model:
        dino=DinoCopy(args.dino_model,device)
        dino.set_train(lambda i,j:bank_to_pixels(bank,X[i:j],vae),X.shape[0],args.bank+"."+args.dino_model.replace("/","_")+".pt")
    rankings=[r for r in args.rankings if r!="dino" or dino is not None]
    os.makedirs(args.output_dir,exist_ok=True)

    def rank_scores(p:int,out:torch.Tensor,pixels:torch.Tensor,attr:dict,seed:int)->dict:
        # a score per bank image for each ranking; deleting the top-k scores is that ranking's counterfactual
        scores={"attr_content":attr["content"][0],"attr_layout":attr["layout"][0]}
        scores["attr"]=(scores["attr_content"]+scores["attr_layout"])/2
        if dino is not None:
            scores["dino"]=(dino.embed(pixels)@dino.train_feats.T)[0].cpu()
        scores["clip"]=(encode_images_clip(clip_model,clip_processor,to_pil(pixels),device)@T.T)[0].cpu()
        chunk=row_chunk(X)
        scores["pixel"]=-torch.cat([torch.cdist(out.float(),X[j:j+chunk].to(device).float())[0].cpu() for j in range(0,X.shape[0],chunk)])
        pool=torch.topk(log_priors[p],min(args.top_k if args.top_k>0 else X.shape[0],X.shape[0])).indices.cpu()
        scores["random"]=torch.full((X.shape[0],),-float("inf"))
        scores["random"][pool]=torch.rand(pool.numel(),generator=torch.Generator().manual_seed(seed))
        return {r:scores[r] for r in rankings}

    def change(out,pixels,cf,cf_pixels)->dict:
        rec={"l2":float((cf-out).norm()/sampler.scale)}
        if lpips is not None:
            with torch.no_grad():
                rec["lpips"]=float(lpips(pixels.to(device),cf_pixels.to(device)).mean())
        if dino is not None:
            rec["dino_sim"]=float((dino.embed(pixels)*dino.embed(cf_pixels)).sum())
        return rec

    def figure_row(out,pixels,attr,scores,p,seed)->object:
        tiles=[pixels]
        if "sources" in attr:
            s=attr["sources"]
            img,ys,xs=s["image"][0,:,:,0],s["y"][0,:,:,0],s["x"][0,:,:,0]
            # each pixel's top source pixel, gathered straight from the bank (not a whole image per pixel)
            src=X.view(-1,*shape)[img.clamp_min(0).flatten().to(X.device),:,ys.flatten().to(X.device),xs.flatten().to(X.device)]
            mosaic=src.to(device).float().T.reshape(1,shape[0],*img.shape)
            tiles.append(bank_to_pixels(bank,mosaic.flatten(1),vae))
            top5=torch.topk(attr["content"][0],5).indices
            colour=torch.full((*img.shape,3),0.5)
            for c,i in enumerate(top5.tolist()):
                colour[img==i]=PALETTE[c]
            tiles.append(colour.permute(2,0,1)[None]*2-1)
        top5=torch.topk(scores["attr"],5).indices
        tiles+=list(bank_to_pixels(bank,X[top5.to(X.device)].float(),vae).split(1))
        for r in ["attr","dino"]:
            if r in scores:
                cf=sampler.generate(p,args.w,1,exclude=torch.topk(scores[r],5).indices,seed=seed)[0]
                tiles.append(bank_to_pixels(bank,cf,vae))
        size=pixels.shape[-2:]
        tiles=[torch.nn.functional.interpolate(t.float().cpu(),size=size,mode="nearest") for t in tiles]
        return concat_images_horizontally(to_pil(torch.cat(tiles)))

    records=[]
    rows=[]
    saved={}
    start=time.time()
    for p,prompt in enumerate(prompts):
        for i in range(args.n_per_prompt):
            seed=args.seed+1000*p+i
            out,_,_,attr=sampler.generate(p,args.w,1,attribution=True,seed=seed)
            pixels=bank_to_pixels(bank,out,vae)
            scores=rank_scores(p,out,pixels,attr,seed)
            base={"prompt":prompt,"sample":i}
            # the full bank again: shares the original's noise, so this must not change anything
            cf=sampler.generate(p,args.w,1,seed=seed)[0]
            records.append({**base,"ranking":"none","k":0,**change(out,pixels,cf,bank_to_pixels(bank,cf,vae))})
            for k in args.ks:
                for r,score in scores.items():
                    exclude=torch.topk(score,k).indices
                    cf=sampler.generate(p,args.w,1,exclude=exclude,seed=seed)[0]
                    records.append({**base,"ranking":r,"k":k,**change(out,pixels,cf,bank_to_pixels(bank,cf,vae)),
                                    "content_removed":float(attr["content"][0][exclude].sum()),
                                    "layout_removed":float(attr["layout"][0][exclude].sum())})
            if i<args.n_show:
                rows.append(figure_row(out,pixels,attr,scores,p,seed))
                saved[f"{p}_{i}"]={k:v for k,v in attr.items() if k in ["sources","images"]}
                saved[f"{p}_{i}"]["top"]=torch.topk(attr["content"][0],20)
            print(f"prompt {p} sample {i}: {time.time()-start:.0f}s, top content shares "
                  f"{[round(v,3) for v in torch.topk(attr['content'][0],5).values.tolist()]}")

    if rows:
        concat_images_vertically(rows).save(os.path.join(args.output_dir,"attribution.png"))
    torch.save(saved,os.path.join(args.output_dir,"attribution_maps.pt"))
    summary=summarize(records,args)
    with open(os.path.join(args.output_dir,"attribution.json"),"w") as f:
        json.dump({"args":vars(args),"prompts":prompts,"records":records,"summary":summary},f,indent=4)
    print(f"saved to {args.output_dir}")


def summarize(records:list,args)->dict:
    '''
    mean and standard error of each change metric per (ranking, k), and for each baseline the fraction of samples
    where deleting our top-k changes the sample more (LPIPS, or l2 without it) than deleting the baseline's
    '''
    metric="lpips" if records and "lpips" in records[0] else "l2"
    keys=["l2","lpips","dino_sim","content_removed","layout_removed"]
    table={}
    for r in ["none"]+list(dict.fromkeys(x["ranking"] for x in records if x["ranking"]!="none")):
        for k in sorted({x["k"] for x in records if x["ranking"]==r}):
            rows=[x for x in records if x["ranking"]==r and x["k"]==k]
            entry={}
            for m in keys:
                vals=np.array([x[m] for x in rows if m in x])
                if len(vals):
                    entry[m]=float(vals.mean())
                    entry[m+"_se"]=float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else 0.0
            table[f"{r} k={k}"]=entry
    wins={}
    by=lambda r,k:{(x["prompt"],x["sample"]):x[metric] for x in records if x["ranking"]==r and x["k"]==k}
    for k in args.ks:
        ours=by("attr",k)
        for b in [r for r in args.rankings if not r.startswith("attr")]:
            theirs=by(b,k)
            common=[s for s in ours if s in theirs]
            if common:
                wins[f"attr vs {b} k={k}"]=float(np.mean([ours[s]>theirs[s] for s in common]))
    lines=["| ranking | "+" | ".join(keys)+" |","|"+"---|"*(len(keys)+1)]
    for name,e in table.items():
        lines.append(f"| {name} | "+" | ".join(f"{e[m]:.4f} ± {e[m+'_se']:.4f}" if m in e else "-" for m in keys)+" |")
    lines+=["",f"fraction of samples where deleting our top-k changes the sample more ({metric}):",""]
    lines+=[f"- {name}: {v:.3f}" for name,v in wins.items()]
    with open(os.path.join(args.output_dir,"summary.md"),"w") as f:
        f.write("\n".join(lines)+"\n")
    print("\n".join(lines))
    return {"table":table,"wins":wins}


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    seconds=time.time()-start
    print(f"successful attribution:) time elapsed: {seconds} seconds = {seconds/3600} hours")
    print("all done!")
