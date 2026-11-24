# the CelebA-HQ experiments, shared by the numbered stage scripts (run them in order, each once the previous
# stage's jobs have finished; 3 only needs 1, so it can run alongside 2):
#   gen_1_build.py          embed -> BLIP captions -> CLIP tokens, one step per setting per run (run it until it
#                           only reports "already built")
#   gen_2_train.py          the text-conditional nets (finetune.py)
#   gen_3_sample_closed.py  the retrieval baseline, the whole-image CFDM and the pyramid variants
#   gen_4_sample_nets.py    the trained nets (needs 2)
#   gen_5_agreement.py      closed-form machines vs the scratch net from the same noise (needs 2)
#   gen_6_attribution.py    exact attribution and leave-k-out counterfactuals (attribution.py; needs 1 only)
#   gen_7_summarize.py      summarize.py over each setting's finished runs (run directly, not through sbatch)
# each prints its jobs: python scripts/gen_<N>_<stage>.py [setting ...] | bash; finished runs are skipped
#
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
# make the log dirs first: mkdir -p slurm_chip/embed slurm_chip/caption slurm_chip/tokens slurm_chip/train slurm_chip/sample slurm_chip/agreement slurm_chip/attribution
# FID/KID need torch-fidelity in the cluster env: pip install torch-fidelity
# SHARPEN=1 python scripts/gen_3_sample_closed.py (or gen_4_sample_nets.py): the sample runs again into
# outputs/<run>_sharp with before/after-unsharp-mask grids (radius 1 px up to 64px output, 2 px above); metrics are
# the same as the plain runs
import os
import re
import sys
import zipfile

CELEBAHQ="--dataset korexyz/celeba-hq-256x256 --test_split validation"
# the token file is built from whichever setting's bank is captioned first (gen_1_build.py)
TOKENS="banks/celebahq_tokens.pt"
SETTINGS={
    "celebahq_pix64":{
        "embed":f"{CELEBAHQ} --space pixel --resolution 64",
        "base":"google/ddpm-celebahq-256",
        "sample_size":64,
        # gradient checkpointing: batch 64 runs out of memory on the 10.5GB GPUs
        "train":"--batch_size 64 --gradient_checkpointing --max_steps 100000",
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
        # the ls machine over 1000 256px images per step is too slow to sample 2000 images for FID
        "ls":False,
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
# CLIP score uses a different CLIP than the one conditioning the samplers (the bank's ViT-L/14)
# 2000 samples (FID is biased upward at small N; fid_std is its bootstrap std), and a text-alignment score from
# outside the CLIP family (BLIP image-text matching)
metrics="--n_per_prompt 4 --fid_samples 2000 --kid_subset 100 --fid_ref_size 2000 --eval_clip_model openai/clip-vit-base-patch16 --itm_model Salesforce/blip-itm-base-coco"
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


def selected_settings()->list:
    # the settings named on the command line, all of them by default
    settings=sys.argv[1:] or list(SETTINGS)
    if any(s not in SETTINGS for s in settings):
        sys.exit(f"usage: python {sys.argv[0]} [setting ...]\n  settings: {' '.join(SETTINGS)}\n  got {settings}")
    return settings


def setting(name:str)->dict:
    '''
    the setting's config plus what every stage derives from it: its bank, its trained nets, the net size flags,
    and (SHARPEN=1) the *_sharp run suffix and unsharp-mask flags
    '''
    cfg=dict(SETTINGS[name])
    nets={"pretrained":f"pretrained:{cfg['base']}","ft":f"weights/{name}_ft/ema","scratch":f"weights/{name}_scratch/ema"}
    size=f"--sample_size {cfg['sample_size']}" if cfg.get("sample_size") else ""
    if cfg.get("channels"):
        size+=f" --channels {cfg['channels']}"
    # SHARPEN=1: separate *_sharp runs (so finished plain runs don't block them) that also show unsharp-masked samples
    sharpen=os.environ.get("SHARPEN","")=="1"
    resolution=int(re.search(r"--resolution (\d+)",cfg["embed"]).group(1)) if "--resolution" in cfg["embed"] else 32
    cfg.update({
        "bank":f"banks/{name}.pt",
        "nets":{k:v for k,v in nets.items() if k in cfg.get("nets",nets)},
        "size":size,
        "sharp_suffix":"_sharp" if sharpen else "",
        "sharp_flags":f"--unsharp_amount 1.0 --unsharp_radius {1.0 if resolution<=64 else 2.0}" if sharpen else "",
    })
    return cfg
