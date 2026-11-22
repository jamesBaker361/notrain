'''
the closed-form samplers over one bank, shared by sample.py and attribution.py:
    retrieval  training images drawn from the w-sharpened prior over the prompt's top-k CLIP matches
    cfdm       smoothed CFDM (cfdm.py) over those matches, at full resolution
    pyramid    that CFDM at the coarsest pyramid level, then closed-form patch levels (pyramid.py); with
               --pyramid_upsample the CFDM layout alone, bicubic-upsampled (what the patch levels add)
    ls         Kamb & Ganguli's local score machine (els.py) over those matches
every sample of retrieval, cfdm and pyramid is an exact convex combination of training data, so generate() can
return its exact attribution, and can run on the bank without some images (exclude): an exact leave-k-out
counterfactual, the same sampler retrained without them
'''
import os
import math
import torch
from cfdm import top_k_subset,sample,median_distance
from pyramid import pyramid_sizes,downsample_flat,build_pyramid_banks,sample_pyramid,resize
from els import machine_sample,patch_schedule_from_fracs,CELEBA_PATCH_FRACS
from bank import encode_texts_clip,tokens_path

METHODS=["retrieval","cfdm","pyramid","ls"]


def add_sampler_args(parser):
    parser.add_argument("--sigma",type=float,default=0.25,help="CFDM smoothing, as a fraction of the median pairwise distance between bank points (at the coarsest level for --method pyramid); 0 reproduces training points exactly")
    parser.add_argument("--M",type=int,default=4,help="noise samples per smoothed score evaluation")
    parser.add_argument("--steps",type=int,default=50,help="CFDM Euler steps (and the net's and the ls machine's sampling steps)")
    parser.add_argument("--start_t",type=float,default=0.2,help="CFDM starts from the exact mixture at this t (0 = pure noise)")
    parser.add_argument("--top_k",type=int,default=1000,help="restrict the CFDM to the top-k CLIP matches per prompt (<=0 for all); w=0 always uses the full bank")
    parser.add_argument("--coarse_size",type=int,default=16,help="pyramid: short side of the coarsest level, where the CFDM runs; levels halve from full resolution down to it")
    parser.add_argument("--patch_size",type=int,default=7,help="pyramid: side of the square patches at every patch level")
    parser.add_argument("--patch_images",type=int,default=50,help="pyramid: images supplying patches, the best CLIP matches (w=0: uniformly random ones); cost grows linearly with it")
    parser.add_argument("--pyramid_steps",type=int,default=10,help="pyramid: denoising steps per patch level")
    parser.add_argument("--pyramid_upsample",action="store_true",help="pyramid ablation: no patch levels, the coarse CFDM sample bicubic-upsampled to full resolution")
    parser.add_argument("--fold_rho",type=float,default=0.2,help="pyramid: width of the Gaussian window overlapping patch estimates are averaged with")
    parser.add_argument("--eta",type=float,default=0.0,help="pyramid: patch-level stochasticity in [0,1] (0 = deterministic DDIM)")
    parser.add_argument("--heatmap_lambda",type=float,default=0.0,help="pyramid: logit bonus per standard deviation of a patch's CLIP text-heatmap score (0 = off)")
    parser.add_argument("--heatmap_keep",type=float,default=1.0,help="pyramid: keep this fraction of CLIP tokens, those with the highest heat over the patch images, and every patch overlapping a kept token (1 = all)")
    parser.add_argument("--heatmap_overlap",type=float,default=0.0,help="pyramid: with --heatmap_keep < 1, the fraction of a patch's pixels that must lie inside kept tokens for it to be kept (0 = any pixel, 0.5 = at least half)")
    parser.add_argument("--heatmap_random",type=float,default=0.0,help="pyramid: with --heatmap_keep < 1, also keep this random fraction of the other patches")
    parser.add_argument("--tokens",type=str,default="",help="pyramid heatmaps: the bank's CLIP token embeddings from clip_tokens.py; defaults to <bank>_tokens.pt")
    parser.add_argument("--heatmap_baseline",type=str,nargs="+",default=["a photo.","an image.","a picture."],help="pyramid: generic texts whose heatmaps, with the other prompts', are averaged and subtracted")
    parser.add_argument("--ls_patch_fracs",type=float,nargs="+",default=None,help="ls: patch sizes from noise to data as fractions of the side (default: Lukoianov et al.'s CelebA-64 fit)")
    parser.add_argument("--ls_chunk",type=int,default=1024,help="ls: training images per chunk (memory)")
    parser.add_argument("--attr_topk",type=int,default=8,help="pyramid attribution: bank patches kept per query patch for the per-pixel sources")
    parser.add_argument("--attr_sources",type=int,default=4,help="pyramid attribution: source pixels kept per output pixel")
    parser.add_argument("--bank_device",type=str,default="auto",choices=["auto","cpu","cuda"],help="where the bank lives; auto keeps banks over 4GB (e.g. 256px pixels) in CPU memory and moves only what's needed")
    parser.add_argument("--seed",type=int,default=123)


def load_bank_tokens(path:str,bank:dict,T:torch.Tensor,device):
    '''
    the bank's CLIP token embeddings (clip_tokens.py), memory-mapped so only the patch images' rows are read;
    possibly shared with banks of the same images in other spaces, so the rows must be the same images in order,
    which their CLIP image embeddings show (up to the precision each bank was embedded in)
    '''
    assert os.path.exists(path),f"heatmaps need the bank's CLIP token embeddings: run clip_tokens.py (missing {path})"
    tokens=torch.load(path,weights_only=False,mmap=True)
    assert tokens["clip_model"]==bank["clip_model"] and tokens["tokens"].shape[0]==bank["X"].shape[0],\
        f"{path} doesn't match the bank ({tokens['clip_model']}, {tokens['tokens'].shape[0]} rows)"
    same=(tokens["T"].float().to(device)*T).sum(-1).min().item()
    assert same>0.98,f"{path} was built from different images than the bank (min CLIP cosine {same:.3f})"
    print(f"CLIP tokens {tuple(tokens['tokens'].shape)} from {path} (last {tokens['n_layers']} layers)")
    return tokens


class ClosedForm:
    def __init__(self,args,bank:dict,X:torch.Tensor,T:torch.Tensor,log_priors:torch.Tensor,text_embeds:torch.Tensor,
                 clip_model,clip_processor,device):
        '''
        X: (N,D) bank points (possibly on the CPU), T: (N,E) their CLIP embeddings, log_priors: (P,N) per prompt
        '''
        assert args.method in METHODS,args.method
        self.args,self.bank,self.X,self.T,self.log_priors,self.device=args,bank,X,T,log_priors,device
        self.shape=bank["shape"]
        if X.device!=torch.device(device) and args.method=="cfdm":
            raise ValueError("whole-bank cfdm needs the bank on the GPU; use --method pyramid or --bank_device cuda")
        # typical distance between bank points: sets the smoothing and normalizes the copy metric
        self.generator=torch.Generator(device=device).manual_seed(args.seed)
        self.scale=median_distance(X,generator=self.generator)
        self.sigma=args.sigma*self.scale
        self.heat_on=False
        if args.method=="pyramid":
            # the whole bank at the coarsest level, once: the CFDM runs there, and its smoothing is set there
            sizes=pyramid_sizes(self.shape[1],self.shape[2],args.coarse_size)
            self.coarse_size=sizes[-1]
            self.X_coarse=downsample_flat(X,self.shape,self.coarse_size,device)
            self.sigma=args.sigma*median_distance(self.X_coarse,generator=self.generator)
            print(f"pyramid levels {sizes} (CFDM at {sizes[-1]}, {args.patch_size}x{args.patch_size} patches above)")
            self.heat_on=args.heatmap_lambda!=0 or args.heatmap_keep<1
            if self.heat_on:
                # every prompt's heatmaps, minus their mean over all prompts and the generic baseline texts
                self.heat_texts=torch.cat([text_embeds,encode_texts_clip(clip_model,clip_processor,args.heatmap_baseline,device)])
                self.tokens=load_bank_tokens(args.tokens or tokens_path(args.bank),bank,T,device)
        if args.method=="ls":
            self.ls_schedule=patch_schedule_from_fracs(args.ls_patch_fracs or CELEBA_PATCH_FRACS,self.shape[-1])
        self._banks={}
        print(f"median pairwise distance {self.scale:.4f}, sigma {self.sigma:.4f}")

    def prior(self,p:int,exclude:torch.Tensor=None)->torch.Tensor:
        # prompt p's log prior over the bank, with the excluded images impossible
        lp=self.log_priors[p]
        if exclude is not None and exclude.numel():
            lp=lp.clone()
            lp[exclude.to(lp.device)]=-math.inf
        return lp

    def patch_banks(self,p:int,w:float,exclude:torch.Tensor=None):
        '''
        the pyramid's patch banks for prompt p at weight w and the (K,) bank rows of their patch images; built
        once and reused (e.g. across FID batches), except on a counterfactual bank (exclude)
        '''
        args=self.args
        if exclude is None and (p,w) in self._banks:
            return self._banks[(p,w)]
        N=self.X.shape[0]
        if w==0:
            # unconditional: random images, no text prior or heatmap (the CLIP matches would themselves be conditioning)
            idx=torch.randperm(N,generator=torch.Generator().manual_seed(args.seed+p))
            if exclude is not None:
                idx=idx[~torch.isin(idx,exclude.cpu())]
            idx=idx[:args.patch_images]
            prior,heat=None,None
        else:
            lp=self.prior(p,exclude)
            idx=torch.topk(lp,min(args.patch_images,N)).indices.cpu()
            prior=lp[idx.to(lp.device)]
            heat=None
            if self.heat_on:
                # (K,g,g,texts) heatmaps on the patch images' token grids; pyramid.py maps each token to the
                # patches overlapping it at every level
                maps=self.tokens["tokens"][idx].to(self.device).float()@self.heat_texts.T
                heat=maps[...,p]-maps.mean(-1)
        images=self.X[idx.to(self.X.device)].to(self.device).float().reshape(-1,*self.shape)
        # no heatmap (w=0) means no filtering either
        keep=args.heatmap_keep if heat is not None else 1.0
        # the random share of --heatmap_random has its own seed, so a counterfactual draws its noise like the original
        bank_gen=torch.Generator(device=self.device).manual_seed(args.seed+p)
        banks=build_pyramid_banks(images,args.coarse_size,args.patch_size,prior,w,heat,args.heatmap_lambda,
                                  keep,args.heatmap_random,bank_gen,args.heatmap_overlap)
        info={"patches_per_level":[b["patches"].shape[0] for b in banks],
              # no patch levels when --coarse_size is the full resolution: the CFDM alone
              "patches_kept":sum(b["patches"].shape[0] for b in banks)/max(sum(b["n_total"] for b in banks),1)}
        if exclude is None:
            print(f"prompt {p} w={w}: patch banks {info}")
            self._banks[(p,w)]=(banks,info,idx)
        return banks,info,idx

    def generate(self,p:int,w:float,n:int,exclude:torch.Tensor=None,attribution:bool=False,seed:int=None):
        '''
        n samples for prompt p at weight w, on the bank without the rows in exclude (None: the whole bank);
        seed restarts the sampling noise (a counterfactual with the same seed shares the original's noise draws)
        returns (samples (n,D), prior_mass, extra metrics, attribution or None):
            attribution["layout"]: (n,N) exact convex weights over the bank behind the sample (for the pyramid,
                the CFDM's at the coarsest level); "content": (n,N) the exact share of each bank image in the
                output pixels (the finest patch level; = layout for retrieval and cfdm); pyramid only: "maps"
                (n,K,H,W) per-pixel shares of the patch images "images" (K,), and "sources" {image (bank row),
                y, x, weight}: (n,H,W,attr_sources) each output pixel's top source pixels
        '''
        args,X=self.args,self.X
        assert not attribution or args.method!="ls","the ls machine has no exact attribution"
        if seed is not None:
            self.generator.manual_seed(seed)
        N=X.shape[0]
        lp=self.prior(p,exclude)
        if w==0:
            # unconditional baseline: the top-k subset would itself be text conditioning
            log_prior,prior_mass=None,1.0
            sub=None if exclude is None else (~torch.isin(torch.arange(N,device=self.device),exclude.to(self.device))).nonzero().squeeze(1)
        else:
            _,log_prior,sub=top_k_subset(self.T,lp,args.top_k)
            # fraction of the w-sharpened prior that the top-k subset keeps
            prior_mass=torch.softmax(w*lp,dim=-1)[sub].sum().item()
            if prior_mass<0.9 and exclude is None:
                print(f"warning: top_k={args.top_k} keeps only {prior_mass:.3f} of the prior mass for prompt {p} at w={w}")

        def dense(weights:torch.Tensor)->torch.Tensor:
            # (n,len(sub)) weights over the subset -> (n,N) over the bank
            if sub is None:
                return weights.float().cpu()
            out=torch.zeros(weights.shape[0],N)
            out[:,sub.cpu()]=weights.float().cpu()
            return out

        if args.method=="retrieval":
            if log_prior is None:
                pick=torch.randint(N if sub is None else sub.numel(),(n,),generator=self.generator,device=self.device)
            else:
                pick=torch.multinomial(torch.softmax(w*log_prior,dim=-1),n,replacement=True,generator=self.generator)
            idx=pick if sub is None else sub[pick]
            out=X[idx.to(X.device)].to(self.device).float()
            attr=None
            if attribution:
                onehot=torch.zeros(n,N)
                onehot[torch.arange(n),idx.cpu()]=1
                attr={"layout":onehot,"content":onehot}
            return out,prior_mass,{},attr
        X_w=X if sub is None else X[sub.to(X.device)]
        if args.method=="ls":
            z0=torch.randn((n,*self.shape),generator=self.generator,device=self.device)
            out=machine_sample("ls",z0,X_w.reshape(-1,*self.shape),self.ls_schedule,args.steps,log_prior,w,chunk=args.ls_chunk)
            return out.flatten(1),prior_mass,{},None
        if args.method=="cfdm":
            out=sample(n,X_w,log_prior,w=w,sigma=self.sigma,M=args.M,steps=args.steps,start_t=args.start_t,
                       generator=self.generator,return_weights=attribution)
            if not attribution:
                return out,prior_mass,{},None
            out,weights=out
            weights=dense(weights)
            return out,prior_mass,{},{"layout":weights,"content":weights}
        coarse=self.X_coarse if sub is None else self.X_coarse[sub]
        if args.pyramid_upsample:
            out=sample(n,coarse,log_prior,w=w,sigma=self.sigma,M=args.M,steps=args.steps,start_t=args.start_t,
                       generator=self.generator,return_weights=attribution)
            out,weights=out if attribution else (out,None)
            out=resize(out.reshape(n,self.shape[0],*self.coarse_size),self.shape[1:]).flatten(1)
            attr=None
            if attribution:
                weights=dense(weights)
                attr={"layout":weights,"content":weights}
            return out,prior_mass,{},attr
        banks,info,idx=self.patch_banks(p,w,exclude)
        out=sample_pyramid(n,coarse,self.shape,banks,args.coarse_size,args.patch_size,log_prior,w=w,sigma=self.sigma,
                           M=args.M,cfdm_steps=args.steps,start_t=args.start_t,steps=args.pyramid_steps,rho=args.fold_rho,
                           eta=args.eta,generator=self.generator,return_attribution=attribution,topk=args.attr_topk,
                           n_sources=args.attr_sources)
        if not attribution:
            return out,prior_mass,info,None
        out,a=out
        attr={"layout":dense(a["layout"])}
        if a["content"] is None:
            attr["content"]=attr["layout"]
        else:
            maps=a["content"]["maps"].float().cpu()
            content=torch.zeros(n,N)
            content[:,idx]=maps.mean((2,3))
            sources={k:v.cpu() for k,v in a["content"]["sources"].items()}
            # local patch-image index -> bank row
            sources["image"]=torch.where(sources["image"]>=0,idx[sources["image"].clamp_min(0)],-1)
            attr.update({"content":content,"maps":maps,"images":idx,"sources":sources})
        return out,prior_mass,info,attr
