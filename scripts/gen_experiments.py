# CLIP-tree closed-form samplers vs trained diffusion models, on standard datasets:
#   cifar10_pix32    uoft-cs/cifar10 (test split held out), pixels 32x32,   nets from google/ddpm-cifar10-32
#   celebahq_pix256  korexyz/celeba-hq-256x256 (validation held out), pixels 256x256, nets from google/ddpm-celebahq-256
#   celebahq_ldm     same data in the VQ latents (3x64x64) of CompVis/ldm-celebahq-256, nets from that LDM
# trained nets per setting: the checkpoint as-is (unconditional), the checkpoint + cross-attention fine-tuned on
# captions ("ft"), and the same architecture + cross-attention trained from scratch on our bank ("scratch")
# usage: python scripts/gen_experiments.py build|train|sample|agreement|summarize [setting ...]
# each stage needs the previous one to finish (sample's analytical runs only need build)
# make the log dirs first: mkdir -p slurm_chip/embed slurm_chip/caption slurm_chip/tree slurm_chip/train slurm_chip/sample slurm_chip/agreement
# FID/KID need torch-fidelity in the cluster env: pip install torch-fidelity
import sys
stage=sys.argv[1] if len(sys.argv)>1 else "all"

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
        # locality/wiener need d x d matrices (d = 196608), and depth 0 would need the whole bank on the GPU
        "methods":["cfdm","ls"],
        "depths":[2,4,8,99],
    },
    "celebahq_ldm":{
        "embed":"--dataset korexyz/celeba-hq-256x256 --test_split validation --space vq --vae CompVis/ldm-celebahq-256 --resolution 256",
        "blip":True,
        "base":"CompVis/ldm-celebahq-256",
        "prompts":"prompts/celeb.txt",
        "train":"--batch_size 16 --gradient_checkpointing --max_steps 50000",
        "methods":["cfdm","locality","wiener","ls"],
        "depths":[0,2,4,8,99],
    },
}
settings=sys.argv[2:] or list(SETTINGS)

branching="silhouette"
# --c_search 1 so the descent only stops at the depth cutoff (or at a leaf); depth 0 = no tree (the paper's setting),
# depth 99 = always a single-image leaf, i.e. a CLIP-guided retrieval baseline
# CLIP score uses a different CLIP than the one conditioning the descent (the bank's ViT-L/14)
metrics="--n_per_prompt 4 --fid_samples 500 --kid_subset 100 --fid_ref_size 2000 --eval_clip_model openai/clip-vit-base-patch16"
method_flags={
    "cfdm":"--node_sampler cfdm --steps 50 --sigma 0.25 --start_t 0.2",
    "locality":"--node_sampler locality --steps 10 --tau 0.02 --wiener_shrinkage 0.5 --wiener_ridge 0",
    "wiener":"--node_sampler wiener --steps 10 --wiener_shrinkage 0.5 --wiener_ridge 0",
    "ls":"--node_sampler ls --steps 10",
}
# fine-tuning starts from the checkpoint (lower lr); scratch training from random init
train_flags={"ft":"--init pretrained --cross_attention --lr 2e-5","scratch":"--init scratch --cross_attention --lr 1e-4"}


def sbatch(job:str,log:str,cmd:str,dependency:str="")->str:
    dep=f"--dependency=afterok:{dependency} " if dependency else ""
    return f"sbatch --parsable {dep}-J {job} --err=slurm_chip/{log}.err --out=slurm_chip/{log}.out runpygpu_chip.sh {cmd}"


for name in settings:
    cfg=SETTINGS[name]
    bank=f"banks/{name}.pt"
    tree=f"banks/{name}_tree.pt"
    nets={"pretrained":f"pretrained:{cfg['base']}","ft":f"weights/{name}_ft/ema","scratch":f"weights/{name}_scratch/ema"}
    if stage in ["build","all"]:
        # embed -> (BLIP captions) -> tree, each waiting for the previous job
        embed_args=cfg["embed"]
        line=f"jid=$({sbatch('embed_'+name,'embed/'+name,f'embed.py {embed_args} --save_path {bank}')})"
        if cfg["blip"]:
            line+=f" && jid=$({sbatch('caption_'+name,'caption/'+name,f'caption.py --bank {bank}','$jid')})"
        line+=f" && {sbatch('tree_'+name,'tree/'+name,f'tree.py --bank {bank} --save_path {tree} --branching {branching}','$jid')}"
        print(line)
    if stage in ["train","all"]:
        for kind,flags in train_flags.items():
            print(sbatch(f"{kind}_{name}",f"train/{name}_{kind}",f"finetune.py --bank {bank} --base {cfg['base']} --name {name}_{kind} {flags} {cfg['train']}"))
    if stage in ["sample","all"]:
        for depth in cfg["depths"]:
            for method in cfg["methods"]:
                if depth==99 and method!="cfdm":
                    # every sampler returns the leaf's image, so one retrieval run is enough
                    continue
                run=f"{name}_{method}_L{depth}" if depth!=99 else f"{name}_retrieval"
                print(sbatch(f"{method}_L{depth}",f"sample/{run}",f"sample.py --bank {bank} --tree {tree} --prompt_file {cfg['prompts']} --method tree --c_search 1 --w 0 1 4 {metrics} --tree_depth {depth} {method_flags[method]} --output_dir outputs/{run}"))
        for kind,spec in nets.items():
            # the as-is checkpoint is unconditional, so only w=0 means anything; the cross-attention nets use w as the CFG scale
            ws="0" if kind=="pretrained" else "0 1 4"
            run=f"{name}_net_{kind}"
            print(sbatch(f"net_{kind}",f"sample/{run}",f"sample.py --bank {bank} --prompt_file {cfg['prompts']} --method net --model {spec} --steps 50 --w {ws} {metrics} --output_dir outputs/{run}"))
    if stage in ["agreement","all"]:
        # target: the net trained from scratch on exactly our bank; the checkpoint and its fine-tune as "another DDPM"
        print(sbatch(f"agree_{name}",f"agreement/{name}",f"agreement.py --bank {bank} --model {nets['scratch']} --other_models {nets['pretrained']} {nets['ft']} --methods {' '.join(['ideal']+cfg['methods'])} --n 64 --output_dir outputs/agreement_{name}"))
    if stage in ["summarize","all"]:
        print(f"python summarize.py --runs 'outputs/{name}_*' --output_dir outputs/summary_{name}")
