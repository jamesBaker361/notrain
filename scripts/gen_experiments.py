# closed-form samplers (whole-image CFDM, and the CFDM -> patch pyramid of pyramid.py) vs trained diffusion models,
# on CelebA-HQ (korexyz/celeba-hq-256x256, validation split held out, BLIP captions) in four spaces:
#   celebahq_pix64   pixels at 64x64 (the locality paper's setting); no 64px checkpoint exists, so the nets are the
#                    celebahq-256 architecture run at 64x64 with cross-attention (from scratch, and fine-tuned from
#                    the 256px weights)
#   celebahq_pix256  pixels at 256x256, nets from google/ddpm-celebahq-256
#   celebahq_ldm     the VQ latents (3x64x64) of CompVis/ldm-celebahq-256, nets from that LDM
#   celebahq_flux2   the normalized 32x32x32 latents of the FLUX.2 autoencoder (black-forest-labs/FLUX.2-dev, gated:
#                    accept its license and log in on the cluster); no checkpoint in that space, so one scratch net
#                    (the ldm-celebahq-256 UNet with 32 channels at 32x32)
# all four share one file of CLIP token embeddings (clip_tokens.py): CLIP sees the original images whatever the space
# trained nets per setting: the checkpoint as-is (unconditional), the checkpoint + cross-attention fine-tuned on
# captions ("ft"), and the same architecture + cross-attention trained from scratch on our bank ("scratch")
# usage: python scripts/gen_experiments.py build|train|sample_closed|sample_nets|agreement [setting ...]
# (summaries of the finished runs: scripts/gen_summarize.py)
# (sample = sample_closed + sample_nets); sample_closed only needs build, sample_nets and agreement need train
# sample_closed: the retrieval baseline, the whole-image CFDM on the prompt's top-K CLIP matches, and each pyramid
# variant (pyramid_variants)
# make the log dirs first: mkdir -p slurm_chip/embed slurm_chip/caption slurm_chip/tokens slurm_chip/train slurm_chip/sample slurm_chip/agreement
# FID/KID need torch-fidelity in the cluster env: pip install torch-fidelity
# SHARPEN=1 python scripts/gen_experiments.py sample_closed ...: the sample runs again into outputs/<run>_sharp with
# before/after-unsharp-mask grids (radius 1 px up to 64px output, 2 px above); metrics are the same as the plain runs
import os
import re
import sys
import zipfile
stage=sys.argv[1] if len(sys.argv)>1 else "all"
sharpen=os.environ.get("SHARPEN","")=="1"

CELEBAHQ="--dataset korexyz/celeba-hq-256x256 --test_split validation"
# the token file is built from the first setting's bank (the 64px one, the smallest)
TOKENS="banks/celebahq_tokens.pt"
SETTINGS={
    "celebahq_pix64":{
        "embed":f"{CELEBAHQ} --space pixel --resolution 64",
        "base":"google/ddpm-celebahq-256",
        "sample_size":64,
        "train":"--batch_size 64 --max_steps 100000",
        # the 256px checkpoint as-is doesn't apply to 64px inputs
        "nets":["ft","scratch"],
    },
    "celebahq_pix256":{
        # 28000 x 196608 pixels: stored in float16 (11GB) and kept in CPU memory by sample.py/agreement.py
        "embed":f"{CELEBAHQ} --space pixel --resolution 256 --store_dtype float16 --batch_size 16",
        "base":"google/ddpm-celebahq-256",
        "train":"--batch_size 8 --gradient_checkpointing --max_steps 50000",
        # whole-bank cfdm would need the 11GB bank on the GPU; the pyramid only moves its patch images there, and
        # fewer of them, since every 256px image gives 62k patches
        "cfdm":False,
        "pyramid":"--coarse_size 16 --patch_size 7 --patch_images 10",
    },
    "celebahq_ldm":{
        "embed":f"{CELEBAHQ} --space vq --vae CompVis/ldm-celebahq-256 --resolution 256",
        "base":"CompVis/ldm-celebahq-256",
        "train":"--batch_size 16 --gradient_checkpointing --max_steps 50000",
        # the LDM UNet (274M parameters) trains on the bigger GPU nodes
        "train_script":"runpygpu_chip_heavy.sh",
    },
    "celebahq_flux2":{
        # bf16: the FLUX.2 VAE upcasts in fp16
        "embed":f"{CELEBAHQ} --space flux2 --resolution 256 --mixed_precision bf16",
        "base":"CompVis/ldm-celebahq-256",
        "sample_size":32,
        "channels":32,
        "train":"--batch_size 16 --gradient_checkpointing --max_steps 50000",
        "train_script":"runpygpu_chip_heavy.sh",
        "nets":["scratch"],
        # 32x32 latents: CFDM at 8x8, 5x5 patches at 16 and 32
        "pyramid":"--coarse_size 8 --patch_size 5",
    },
}
for cfg in SETTINGS.values():
    cfg.update({"blip":True,"prompts":"prompts/celeb.txt","tokens":TOKENS})
settings=sys.argv[2:] or list(SETTINGS)
STAGES=["build","train","sample","sample_closed","sample_nets","agreement","all"]
if stage not in STAGES or any(s not in SETTINGS for s in settings):
    sys.exit(f"usage: python scripts/gen_experiments.py <stage> [setting ...]\n  stages: {' '.join(STAGES)}\n  settings: {' '.join(SETTINGS)}\n  got stage={stage!r} settings={settings}")

# CLIP score uses a different CLIP than the one conditioning the samplers (the bank's ViT-L/14)
metrics="--n_per_prompt 4 --fid_samples 500 --kid_subset 100 --fid_ref_size 2000 --eval_clip_model openai/clip-vit-base-patch16"
# the CFDM (whole image, or the pyramid's coarsest level) runs over the prompt's top-1000 CLIP matches
cfdm_flags="--top_k 1000 --steps 50 --sigma 0.25 --start_t 0.2"
# pyramid runs: no heatmap; heatmap bonus only; hard filter to the top quarter (expected to tile the prompt's
# regions everywhere); top quarter plus a random quarter of the rest; that filter with patches kept only when at
# least half their pixels are in kept tokens (any overlapping pixel otherwise keeps most patches); and the
# filter plus the bonus
pyramid_variants={
    "pyramid":"",
    "pyramid_hm":"--heatmap_lambda 1",
    "pyramid_strict":"--heatmap_keep 0.25",
    "pyramid_keep":"--heatmap_keep 0.25 --heatmap_random 0.25",
    "pyramid_keep_half":"--heatmap_keep 0.25 --heatmap_random 0.25 --heatmap_overlap 0.5",
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


def sbatch(job:str,log:str,cmd:str,script:str="runpygpu_chip.sh")->str:
    return f"sbatch -J {job} --err=slurm_chip/{log}.err --out=slurm_chip/{log}.out {script} {cmd}"


# shared token files already submitted by an earlier setting in this run
tokens_queued=set()
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
        # embed -> (BLIP captions) -> CLIP token embeddings (clip_tokens.py, for the pyramid's heatmaps): only the
        # next missing step is submitted (caption.py rewrites the bank, so nothing else may read it meanwhile);
        # re-run build once those jobs are done to submit the step after
        # (jobs still queued or running aren't detected: don't re-run build until they're done)
        captioned=False if not os.path.exists(bank) else has_captions(bank)
        if not os.path.exists(bank):
            print(sbatch("embed_"+name,"embed/"+name,f"embed.py {cfg['embed']} --save_path {bank}"))
        elif captioned is None:
            print(f"# {name}: {bank} can't be read (still being written?), skipping its build",file=sys.stderr)
        elif cfg["blip"] and not captioned:
            print(sbatch("caption_"+name,"caption/"+name,f"caption.py --bank {bank} {cfg.get('caption','')}"))
        elif not os.path.exists(cfg["tokens"]) and cfg["tokens"] not in tokens_queued:
            # one shared token file per image set, from whichever of its banks is ready first
            tokens_queued.add(cfg["tokens"])
            log=os.path.basename(cfg["tokens"])[:-3]
            print(sbatch("tokens_"+name,"tokens/"+log,f"clip_tokens.py --bank {bank} --save_path {cfg['tokens']}"))
        elif os.path.exists(cfg["tokens"]):
            print(f"# {name}: bank, captions and CLIP tokens already built",file=sys.stderr)
    if stage in ["train","all"]:
        for kind,flags in train_flags.items():
            if kind not in nets:
                continue
            print(sbatch(f"{kind}_{name}",f"train/{name}_{kind}",f"finetune.py --bank {bank} --base {cfg['base']} --name {name}_{kind} {flags} {size} {cfg['train']}",
                         script=cfg.get("train_script","runpygpu_chip.sh")))
    if stage in ["sample","sample_closed","all"]:
        # the baseline: training images drawn from the prompt's top matches (what sigma -> 0 collapses to)
        runs={"retrieval":"--method retrieval --top_k 1000"}
        if cfg.get("cfdm",True):
            runs["cfdm"]=f"--method cfdm {cfdm_flags}"
        for variant,flags in pyramid_variants.items():
            runs[variant]=f"--method pyramid {cfdm_flags} --pyramid_steps 10 {cfg.get('pyramid','')} --tokens {cfg['tokens']} {flags}"
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
