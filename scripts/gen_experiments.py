# closed-form samplers (whole-image CFDM, and the CFDM -> patch pyramid of pyramid.py) vs trained diffusion models,
# on standard datasets:
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
# usage: python scripts/gen_experiments.py build|train|sample_closed|sample_nets|agreement|summarize [setting ...]
# (sample = sample_closed + sample_nets); sample_closed only needs build, sample_nets and agreement need train
# sample_closed: the whole-image CFDM on the prompt's top-K CLIP matches, and each pyramid variant (pyramid_variants)
# make the log dirs first: mkdir -p slurm_chip/embed slurm_chip/caption slurm_chip/train slurm_chip/sample slurm_chip/agreement
# FID/KID need torch-fidelity in the cluster env: pip install torch-fidelity
# SHARPEN=1 python scripts/gen_experiments.py sample_closed ...: the sample runs again into outputs/<run>_sharp with
# before/after-unsharp-mask grids (radius 1 px up to 64px output, 2 px above); metrics are the same as the plain runs
import os
import re
import sys
import zipfile
stage=sys.argv[1] if len(sys.argv)>1 else "all"
sharpen=os.environ.get("SHARPEN","")=="1"

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
        # 8 -> 16 -> 32: CFDM at 8x8, 5x5 patches at 16 and 32
        "pyramid":"--coarse_size 8 --patch_size 5",
    },
    "celebahq_pix256":{
        # 28000 x 196608 pixels: stored in float16 (11GB) and kept in CPU memory by sample.py/agreement.py
        "embed":"--dataset korexyz/celeba-hq-256x256 --test_split validation --space pixel --resolution 256 --store_dtype float16 --batch_size 16",
        "blip":True,
        "base":"google/ddpm-celebahq-256",
        "prompts":"prompts/celeb.txt",
        "train":"--batch_size 8 --gradient_checkpointing --max_steps 50000",
        # whole-bank cfdm would need the 11GB bank on the GPU; the pyramid only moves its patch images there, and
        # fewer of them, since every 256px image gives 62k patches
        "cfdm":False,
        "pyramid":"--coarse_size 16 --patch_size 7 --patch_images 10",
    },
    "celebahq_ldm":{
        "embed":"--dataset korexyz/celeba-hq-256x256 --test_split validation --space vq --vae CompVis/ldm-celebahq-256 --resolution 256",
        "blip":True,
        "base":"CompVis/ldm-celebahq-256",
        "prompts":"prompts/celeb.txt",
        "train":"--batch_size 16 --gradient_checkpointing --max_steps 50000",
        # the LDM UNet (274M parameters) trains on the bigger GPU nodes
        "train_script":"runpygpu_chip_heavy.sh",
    },
    "celebahq_pix64":{
        "embed":"--dataset korexyz/celeba-hq-256x256 --test_split validation --space pixel --resolution 64",
        "blip":True,
        "base":"google/ddpm-celebahq-256",
        "sample_size":64,
        "prompts":"prompts/celeb.txt",
        "train":"--batch_size 64 --max_steps 100000",
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
        # 4x32x32 latents: CFDM at 8x8, 5x5 patches at 16 and 32, as for CIFAR-10's pixels
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
            **({"pyramid":"--coarse_size 8 --patch_size 5"} if space=="ldm" else {}),
        }
settings=sys.argv[2:] or list(SETTINGS)
STAGES=["build","train","sample","sample_closed","sample_nets","agreement","summarize","all"]
if stage not in STAGES or any(s not in SETTINGS for s in settings):
    sys.exit(f"usage: python scripts/gen_experiments.py <stage> [setting ...]\n  stages: {' '.join(STAGES)}\n  settings: {' '.join(SETTINGS)}\n  got stage={stage!r} settings={settings}")

# CLIP score uses a different CLIP than the one conditioning the samplers (the bank's ViT-L/14)
metrics="--n_per_prompt 4 --fid_samples 500 --kid_subset 100 --fid_ref_size 2000 --eval_clip_model openai/clip-vit-base-patch16"
# the CFDM (whole image, or the pyramid's coarsest level) runs over the prompt's top-1000 CLIP matches
cfdm_flags="--top_k 1000 --steps 50 --sigma 0.25 --start_t 0.2"
# pyramid runs: no heatmap; heatmap bonus only; hard filter to the top quarter (expected to tile the prompt's
# regions everywhere); top quarter plus a random quarter of the rest; and that filter plus the bonus
pyramid_variants={
    "pyramid":"",
    "pyramid_hm":"--heatmap_lambda 1",
    "pyramid_strict":"--heatmap_keep 0.25",
    "pyramid_keep":"--heatmap_keep 0.25 --heatmap_random 0.25",
    "pyramid_keep_hm":"--heatmap_keep 0.25 --heatmap_random 0.25 --heatmap_lambda 1",
}
# fine-tuning starts from the checkpoint (lower lr); scratch training from random init
train_flags={"ft":"--init pretrained --cross_attention --lr 2e-5","scratch":"--init scratch --cross_attention --lr 1e-4"}


def finished(run_dir:str,result:str="metrics.json")->bool:
    # runs with results already on disk are not resubmitted (skipped silently), so re-running a stage only submits what's missing
    return os.path.exists(os.path.join(run_dir,result))


def has_captions(bank:str):
    '''
    whether a bank file already has caption.py's captions, without loading it (pixel banks are >10GB): the pickle
    inside torch.save's zip names every dict key; None if the file can't be read (e.g. still being written)
    '''
    try:
        with zipfile.ZipFile(bank) as z:
            pkl=next(n for n in z.namelist() if n.endswith("data.pkl"))
            data=z.read(pkl)
        # the exact string "captions" (protocol 2 BINUNICODE, or SHORT_BINUNICODE in newer protocols)
        return b"X\x08\x00\x00\x00captions" in data or b"\x8c\x08captions" in data
    except (zipfile.BadZipFile,StopIteration,OSError):
        return None


def sbatch(job:str,log:str,cmd:str,dependency:str="",script:str="runpygpu_chip.sh")->str:
    dep=f"--dependency=afterok:{dependency} " if dependency else ""
    return f"sbatch --parsable {dep}-J {job} --err=slurm_chip/{log}.err --out=slurm_chip/{log}.out {script} {cmd}"


for name in settings:
    cfg=SETTINGS[name]
    bank=f"banks/{name}.pt"
    nets={"pretrained":f"pretrained:{cfg['base']}","ft":f"weights/{name}_ft/ema","scratch":f"weights/{name}_scratch/ema"}
    nets={k:v for k,v in nets.items() if k in cfg.get("nets",nets)}
    size=f"--sample_size {cfg['sample_size']}" if cfg.get("sample_size") else ""
    # SHARPEN=1: separate *_sharp runs (so finished plain runs don't block them) that also show unsharp-masked samples
    resolution=int(re.search(r"--resolution (\d+)",cfg["embed"]).group(1)) if "--resolution" in cfg["embed"] else 32
    sharp_suffix="_sharp" if sharpen else ""
    sharp_flags=f"--unsharp_amount 1.0 --unsharp_radius {1.0 if resolution<=64 else 2.0}" if sharpen else ""
    if cfg.get("channels"):
        size+=f" --channels {cfg['channels']}"
    if stage in ["build","all"]:
        # embed -> (BLIP captions), the captions waiting for the bank; steps whose output is already on disk are
        # skipped (an existing bank isn't re-embedded, a captioned one isn't re-captioned), so re-running build
        # only submits what's missing
        # (jobs still queued or running aren't detected: don't re-run build until they're done)
        need_embed=not os.path.exists(bank)
        captioned=False if need_embed else has_captions(bank)
        jobs=[]
        if captioned is None:
            print(f"# {name}: {bank} can't be read (still being written?), skipping its build",file=sys.stderr)
        elif need_embed:
            jobs.append(("embed_"+name,"embed/"+name,f"embed.py {cfg['embed']} --save_path {bank}"))
        if cfg["blip"] and captioned is False:
            jobs.append(("caption_"+name,"caption/"+name,f"caption.py --bank {bank} {cfg.get('caption','')}"))
        if captioned is not None and not jobs:
            print(f"# {name}: bank and captions already built",file=sys.stderr)
        parts=[]
        dep=""
        for job,log,cmd in jobs:
            parts.append(f"jid=$({sbatch(job,log,cmd,dep)})")
            dep="$jid"
        if parts:
            print(" && ".join(parts))
    if stage in ["train","all"]:
        for kind,flags in train_flags.items():
            if kind not in nets:
                continue
            print(sbatch(f"{kind}_{name}",f"train/{name}_{kind}",f"finetune.py --bank {bank} --base {cfg['base']} --name {name}_{kind} {flags} {size} {cfg['train']}",
                         script=cfg.get("train_script","runpygpu_chip.sh")))
    if stage in ["sample","sample_closed","all"]:
        runs={}
        if cfg.get("cfdm",True):
            runs["cfdm"]=f"--method cfdm {cfdm_flags}"
        for variant,flags in pyramid_variants.items():
            runs[variant]=f"--method pyramid {cfdm_flags} --pyramid_steps 10 {cfg.get('pyramid','')} {flags}"
        for label,flags in runs.items():
            run=f"{name}_{label}{sharp_suffix}"
            if finished(f"outputs/{run}"):
                continue
            print(sbatch(label,f"sample/{run}",f"sample.py --bank {bank} --prompt_file {cfg['prompts']} --w 0 1 4 {metrics} {flags} {sharp_flags} --output_dir outputs/{run}"))
    if stage in ["sample","sample_nets","all"]:
        for kind,spec in nets.items():
            # the as-is checkpoint is unconditional, so only w=0 means anything; the cross-attention nets use w as the CFG scale
            ws="0" if kind=="pretrained" else "0 1 4"
            run=f"{name}_net_{kind}{sharp_suffix}"
            if finished(f"outputs/{run}"):
                continue
            print(sbatch(f"net_{kind}",f"sample/{run}",f"sample.py --bank {bank} --prompt_file {cfg['prompts']} --method net --model {spec} --steps 50 --w {ws} {metrics} {sharp_flags} --output_dir outputs/{run}"))
    if stage in ["agreement","all"] and not finished(f"outputs/agreement_{name}","agreement.json"):
        # target: the net trained from scratch on exactly our bank; the checkpoint and its fine-tune as "another DDPM"
        others=" ".join(v for k,v in nets.items() if k!="scratch")
        print(sbatch(f"agree_{name}",f"agreement/{name}",f"agreement.py --bank {bank} --model {nets['scratch']} --other_models {others} --methods ideal cfdm ls --n 64 --output_dir outputs/agreement_{name}"))
    if stage in ["summarize","all"]:
        print(f"python summarize.py --runs 'outputs/{name}_*' --output_dir outputs/summary_{name}")
