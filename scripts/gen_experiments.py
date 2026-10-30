# CLIP-tree closed-form samplers vs trained diffusion models, on standard datasets:
#   cifar10_pix32    uoft-cs/cifar10 (test split held out), pixels 32x32,   nets from google/ddpm-cifar10-32
#   celebahq_pix256  korexyz/celeba-hq-256x256 (validation held out), pixels 256x256, nets from google/ddpm-celebahq-256
#   celebahq_ldm     same data in the VQ latents (3x64x64) of CompVis/ldm-celebahq-256, nets from that LDM
#   celebahq_pix64   same faces as pixels at 64x64 (the locality paper's setting); no 64px checkpoint exists, so the
#                    nets are the celebahq-256 architecture run at 64x64 with cross-attention (from scratch, and
#                    fine-tuned from the 256px weights)
#   imagenet_pix64   30k shuffled images of benjamin-paine/imagenet-1k-64x64 (2k of validation held out), BLIP captions
#   wikiart_pix64    the first 20 of huggan/wikiart's 72 (already shuffled) shards, ~22k paintings (1k held out),
#                    captions = BLIP + style, genre and artist
#   league_pix64     jlbaker361/league-splash-tagged, 2000 splash arts (200 held out), captions = BLIP + skin and champion
#   these three have no pretrained checkpoint, so their nets are the celebahq-256 architecture at 64x64 with
#   cross-attention, trained from scratch
#   {imagenet,wikiart,league}_pix256 and _ldm: the same data at 256x256 pixels, and in the 4x32x32 latents of the
#   Stable Diffusion VAE (stabilityai/sd-vae-ft-mse; the CelebA-HQ LDM's VQ autoencoder was trained only on faces),
#   set up like celebahq_pix256 / celebahq_ldm with scratch nets only (the ldm-celebahq-256 UNet with 4 channels at
#   32x32). ImageNet at 256 comes from benjamin-paine/imagenet-1k-256x256's validation split (0.7GB, not
#   class-sorted; 30k bank + 2k held out) rather than its 20GB train split
# trained nets per setting: the checkpoint as-is (unconditional), the checkpoint + cross-attention fine-tuned on
# captions ("ft"), and the same architecture + cross-attention trained from scratch on our bank ("scratch")
# usage: python scripts/gen_experiments.py build|tree|train|sample_tree|sample_topk|sample_nets|agreement|summarize [setting ...]
# (tree = only the tree jobs, for banks that already exist, e.g. after adding to cluster_methods)
# (sample = sample_tree + sample_topk + sample_nets); sample_tree and sample_topk only need build, sample_nets and agreement need train
# sample_topk: no tree, each sampler on the prompt's top-K CLIP matches (K in topk_values), weighted by the text prior
# make the log dirs first: mkdir -p slurm_chip/embed slurm_chip/caption slurm_chip/tree slurm_chip/train slurm_chip/sample slurm_chip/agreement
# FID/KID need torch-fidelity in the cluster env: pip install torch-fidelity
import os
import sys
stage=sys.argv[1] if len(sys.argv)>1 else "all"

WIKIART_DATA="--dataset huggan/wikiart --data_files 'data/train-0000*.parquet' 'data/train-0001*.parquet' --holdout 1000 --metadata_columns artist genre style"
WIKIART_CAPTION="--caption_template '{blip}, {style} {genre} by {artist}'"
LEAGUE_DATA="--dataset jlbaker361/league-splash-tagged --shuffle --holdout 200 --metadata_columns champion tag"
LEAGUE_CAPTION="--caption_template '{blip}, {tag} {champion} splash art'"
SETTINGS={
    "cifar10_pix32":{
        "embed":"--dataset uoft-cs/cifar10 --image_column img --test_split test --label_column label --space pixel --resolution 32",
        "blip":False,
        "base":"google/ddpm-cifar10-32",
        "prompts":"prompts/cifar10.txt",
        "train":"--batch_size 128 --max_steps 50000",
        "methods":["cfdm","locality","wiener","ls"],
        "depths":[0,2,4,8,99],
    },
    "celebahq_pix256":{
        # 28000 x 196608 pixels: stored in float16 (11GB) and kept in CPU memory by sample.py/agreement.py
        "embed":"--dataset korexyz/celeba-hq-256x256 --test_split validation --space pixel --resolution 256 --store_dtype float16 --batch_size 16",
        "blip":True,
        "base":"google/ddpm-celebahq-256",
        "prompts":"prompts/celeb.txt",
        "train":"--batch_size 8 --gradient_checkpointing --max_steps 50000",
        # locality needs d x d matrices (d = 196608), so wiener runs low-rank; depth 0 would need the whole bank on the GPU
        "methods":["cfdm","wiener","ls"],
        "depths":[2,4,8,99],
        "extra":{"wiener":"--wiener_rank 512"},
    },
    "celebahq_ldm":{
        "embed":"--dataset korexyz/celeba-hq-256x256 --test_split validation --space vq --vae CompVis/ldm-celebahq-256 --resolution 256",
        "blip":True,
        "base":"CompVis/ldm-celebahq-256",
        "prompts":"prompts/celeb.txt",
        "train":"--batch_size 16 --gradient_checkpointing --max_steps 50000",
        # the LDM UNet (274M parameters) trains on the bigger GPU nodes
        "train_script":"runpygpu_chip_heavy.sh",
        "methods":["cfdm","locality","wiener","ls"],
        "depths":[0,2,4,8,99],
    },
    "celebahq_pix64":{
        "embed":"--dataset korexyz/celeba-hq-256x256 --test_split validation --space pixel --resolution 64",
        "blip":True,
        "base":"google/ddpm-celebahq-256",
        "sample_size":64,
        "prompts":"prompts/celeb.txt",
        "train":"--batch_size 64 --max_steps 100000",
        "methods":["cfdm","locality","wiener","ls"],
        "depths":[0,2,4,8,99],
        # the 256px checkpoint as-is doesn't apply to 64px inputs
        "nets":["ft","scratch"],
    },
    "imagenet_pix64":{
        # sorted by class, hence --shuffle; captions are BLIP's (the class names are kept as metadata "label")
        "embed":"--dataset benjamin-paine/imagenet-1k-64x64 --test_split validation --shuffle --limit 30000 --test_limit 2000 --metadata_columns label --space pixel --resolution 64",
        "blip":True,
        "base":"google/ddpm-celebahq-256",
        "sample_size":64,
        "prompts":"prompts/imagenet.txt",
        "train":"--batch_size 64 --max_steps 100000",
        "methods":["cfdm","locality","wiener","ls"],
        "depths":[0,2,4,8,99],
        "nets":["scratch"],
    },
    "wikiart_pix64":{
        # 20 shards ~ 9GB of the 34GB dataset; only a train split, so the last 1000 are held out
        "embed":f"{WIKIART_DATA} --space pixel --resolution 64",
        "blip":True,
        "caption":WIKIART_CAPTION,
        "base":"google/ddpm-celebahq-256",
        "sample_size":64,
        "prompts":"prompts/wikiart.txt",
        "train":"--batch_size 64 --max_steps 100000",
        "methods":["cfdm","locality","wiener","ls"],
        "depths":[0,2,4,8,99],
        "nets":["scratch"],
    },
    "league_pix64":{
        # sorted by champion, hence --shuffle before the holdout; 1215x717 splashes are center-cropped to squares
        "embed":f"{LEAGUE_DATA} --space pixel --resolution 64",
        "blip":True,
        "caption":LEAGUE_CAPTION,
        "base":"google/ddpm-celebahq-256",
        "sample_size":64,
        "prompts":"prompts/league-splash-tagged.txt",
        # only 1800 training images
        "train":"--batch_size 64 --max_steps 20000",
        "methods":["cfdm","locality","wiener","ls"],
        "depths":[0,2,4,8,99],
        "nets":["scratch"],
    },
}
# the same three datasets at 256x256 pixels and in the LDM's VQ latents, configured like the CelebA-HQ settings
for data,flags,caption,steps in [
    ("imagenet","--dataset benjamin-paine/imagenet-1k-256x256 --data_files 'data/validation-*.parquet' --limit 32000 --holdout 2000","",50000),
    ("wikiart",WIKIART_DATA,WIKIART_CAPTION,50000),
    # only 1800 training images
    ("league",LEAGUE_DATA,LEAGUE_CAPTION,20000),
]:
    for space,template,embed_flags,net_flags in [
        ("pix256","celebahq_pix256","--space pixel --resolution 256 --store_dtype float16 --batch_size 16",{}),
        ("ldm","celebahq_ldm","--space vae --vae stabilityai/sd-vae-ft-mse --resolution 256",{"sample_size":32,"channels":4}),
    ]:
        base=SETTINGS[template]
        SETTINGS[f"{data}_{space}"]={
            **base,
            "embed":f"{flags} {embed_flags}",
            "caption":caption,
            "prompts":f"prompts/{'league-splash-tagged' if data=='league' else data}.txt",
            "train":base["train"].replace("--max_steps 50000",f"--max_steps {steps}"),
            "nets":["scratch"],
            **net_flags,
        }
# every setting also runs the curve sampler (a polyline through the node's images in TSP order, sampled along)
for cfg in SETTINGS.values():
    cfg["methods"]=cfg["methods"]+["curve"]
settings=sys.argv[2:] or list(SETTINGS)
STAGES=["build","tree","train","sample","sample_tree","sample_topk","sample_nets","agreement","summarize","all"]
if stage not in STAGES or any(s not in SETTINGS for s in settings):
    sys.exit(f"usage: python scripts/gen_experiments.py <stage> [setting ...]\n  stages: {' '.join(STAGES)}\n  settings: {' '.join(SETTINGS)}\n  got stage={stage!r} settings={settings}")

# top-K set sizes for sample_topk (the tree-free alternative to the depth cutoff)
topk_values=[50,200,1000]
# trees to build and sample; kmeans keeps the original file/run names, others get a _<name> suffix
# *_caption trees cluster the CLIP text embeddings of the captions instead of the images, *_mix a 50/50 blend;
# CIFAR-10's captions are one template per class, so caption trees there are only 10 real clusters (use mix)
cluster_methods=["dbscan","agglomerative"]
cluster_flags={
    "kmeans":"--cluster_method kmeans --branching silhouette",
    "dbscan":"--cluster_method dbscan --dbscan_min_samples 5 --dbscan_quantile 0.5",
    "agglomerative":"--cluster_method agglomerative --linkage ward",
}
for m in list(cluster_flags):
    cluster_flags[f"{m}_caption"]=cluster_flags[m]+" --cluster_on caption"
    cluster_flags[f"{m}_mix"]=cluster_flags[m]+" --cluster_on mix --caption_weight 0.5"
# --c_search 1 so the descent only stops at the depth cutoff (or at a leaf); depth 0 = no tree (the paper's setting),
# depth 99 = always a single-image leaf, i.e. a CLIP-guided retrieval baseline
# CLIP score uses a different CLIP than the one conditioning the descent (the bank's ViT-L/14)
metrics="--n_per_prompt 4 --fid_samples 500 --kid_subset 100 --fid_ref_size 2000 --eval_clip_model openai/clip-vit-base-patch16"
method_flags={
    "cfdm":"--node_sampler cfdm --steps 50 --sigma 0.25 --start_t 0.2",
    "locality":"--node_sampler locality --steps 10 --tau 0.02 --wiener_shrinkage 0.5 --wiener_ridge 0",
    "wiener":"--node_sampler wiener --steps 10 --wiener_shrinkage 0.5 --wiener_ridge 0",
    "ls":"--node_sampler ls --steps 10",
    "curve":"--node_sampler curve --curve_mode random",
}
# fine-tuning starts from the checkpoint (lower lr); scratch training from random init
train_flags={"ft":"--init pretrained --cross_attention --lr 2e-5","scratch":"--init scratch --cross_attention --lr 1e-4"}


def finished(run_dir:str,result:str="metrics.json")->bool:
    # runs with results already on disk are not resubmitted (skipped silently), so re-running a stage only submits what's missing
    return os.path.exists(os.path.join(run_dir,result))


def sbatch(job:str,log:str,cmd:str,dependency:str="",script:str="runpygpu_chip.sh")->str:
    dep=f"--dependency=afterok:{dependency} " if dependency else ""
    return f"sbatch --parsable {dep}-J {job} --err=slurm_chip/{log}.err --out=slurm_chip/{log}.out {script} {cmd}"


for name in settings:
    cfg=SETTINGS[name]
    bank=f"banks/{name}.pt"
    trees={m:(f"banks/{name}_tree.pt" if m=="kmeans" else f"banks/{name}_tree_{m}.pt") for m in cluster_methods}
    tree_jobs=[(f"tree_{name}_{m}",f"tree/{name}_{m}",f"tree.py --bank {bank} --save_path {path} {cluster_flags[m]}") for m,path in trees.items()]
    nets={"pretrained":f"pretrained:{cfg['base']}","ft":f"weights/{name}_ft/ema","scratch":f"weights/{name}_scratch/ema"}
    nets={k:v for k,v in nets.items() if k in cfg.get("nets",nets)}
    size=f"--sample_size {cfg['sample_size']}" if cfg.get("sample_size") else ""
    if cfg.get("channels"):
        size+=f" --channels {cfg['channels']}"
    if stage in ["build","all"]:
        # embed -> (BLIP captions) -> tree, each waiting for the previous job
        embed_args=cfg["embed"]
        line=f"jid=$({sbatch('embed_'+name,'embed/'+name,f'embed.py {embed_args} --save_path {bank}')})"
        if cfg["blip"]:
            caption_cmd=f"caption.py --bank {bank} {cfg.get('caption','')}"
            line+=f" && jid=$({sbatch('caption_'+name,'caption/'+name,caption_cmd,'$jid')})"
        for job,log,cmd in tree_jobs:
            line+=f" && {sbatch(job,log,cmd,'$jid')}"
        print(line)
    if stage=="tree":
        # trees only, for a bank that already exists (e.g. adding a clustering method)
        for job,log,cmd in tree_jobs:
            print(sbatch(job,log,cmd))
    if stage in ["train","all"]:
        for kind,flags in train_flags.items():
            if kind not in nets:
                continue
            print(sbatch(f"{kind}_{name}",f"train/{name}_{kind}",f"finetune.py --bank {bank} --base {cfg['base']} --name {name}_{kind} {flags} {size} {cfg['train']}",
                         script=cfg.get("train_script","runpygpu_chip.sh")))
    if stage in ["sample","sample_tree","all"]:
        for cm,tree in trees.items():
            suffix="" if cm=="kmeans" else f"_{cm}"
            for depth in cfg["depths"]:
                if depth==0 and cm!=cluster_methods[0]:
                    # depth 0 is the whole bank whatever the tree, so it only runs once
                    continue
                for method in cfg["methods"]:
                    if depth==99 and method!="cfdm":
                        # every sampler returns the leaf's image, so one retrieval run is enough
                        continue
                    run=f"{name}_{method}_L{depth}{suffix}" if depth!=99 else f"{name}_retrieval{suffix}"
                    if finished(f"outputs/{run}"):
                        continue
                    extra=cfg.get("extra",{}).get(method,"")
                    print(sbatch(f"{method}_L{depth}{suffix}",f"sample/{run}",f"sample.py --bank {bank} --tree {tree} --prompt_file {cfg['prompts']} --method tree --c_search 1 --w 0 1 4 {metrics} --tree_depth {depth} {method_flags[method]} {extra} --output_dir outputs/{run}"))
    if stage in ["sample","sample_topk","all"]:
        for k in topk_values:
            for method in cfg["methods"]:
                run=f"{name}_{method}_K{k}"
                if finished(f"outputs/{run}"):
                    continue
                extra=cfg.get("extra",{}).get(method,"")
                print(sbatch(f"{method}_K{k}",f"sample/{run}",f"sample.py --bank {bank} --prompt_file {cfg['prompts']} --method topk --top_k {k} --w 0 1 4 {metrics} {method_flags[method]} {extra} --output_dir outputs/{run}"))
    if stage in ["sample","sample_nets","all"]:
        for kind,spec in nets.items():
            # the as-is checkpoint is unconditional, so only w=0 means anything; the cross-attention nets use w as the CFG scale
            ws="0" if kind=="pretrained" else "0 1 4"
            run=f"{name}_net_{kind}"
            if finished(f"outputs/{run}"):
                continue
            print(sbatch(f"net_{kind}",f"sample/{run}",f"sample.py --bank {bank} --prompt_file {cfg['prompts']} --method net --model {spec} --steps 50 --w {ws} {metrics} --output_dir outputs/{run}"))
    if stage in ["agreement","all"] and not finished(f"outputs/agreement_{name}","agreement.json"):
        # target: the net trained from scratch on exactly our bank; the checkpoint and its fine-tune as "another DDPM"
        others=" ".join(v for k,v in nets.items() if k!="scratch")
        print(sbatch(f"agree_{name}",f"agreement/{name}",f"agreement.py --bank {bank} --model {nets['scratch']} --other_models {others} --methods {' '.join(['ideal']+[m for m in cfg['methods'] if m!='curve'])} {cfg.get('extra',{}).get('wiener','')} --n 64 --output_dir outputs/agreement_{name}"))
    if stage in ["summarize","all"]:
        print(f"python summarize.py --runs 'outputs/{name}_*' --output_dir outputs/summary_{name}")
