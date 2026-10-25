# mechanistic pilot: each stage needs the previous one to finish (bank -> captions -> trained nets -> comparisons)
# make the log dirs first: mkdir -p slurm_chip/caption slurm_chip/train slurm_chip/compare
data="celeb"
stage=__import__("sys").argv[1] if len(__import__("sys").argv)>1 else "all"

if stage in ["caption","all"]:
    print(f"sbatch -J caption --err=slurm_chip/caption/{data}.err --out=slurm_chip/caption/{data}.out runpygpu_chip.sh caption.py --bank banks/{data}.pt")

if stage in ["train","all"]:
    for variant,flag in [("conv",""),("attn","--attention")]:
        print(f"sbatch -J train_{variant} --err=slurm_chip/train/{data}_{variant}.err --out=slurm_chip/train/{data}_{variant}.out runpygpu_chip.sh train.py --bank banks/{data}.pt --name {data}_{variant} {flag}")

if stage in ["compare","all"]:
    for variant in ["conv","attn"]:
        for prior in ["caption","image"]:
            print(f"sbatch -J cmp_{variant}_{prior} --err=slurm_chip/compare/{data}_{variant}_{prior}.err --out=slurm_chip/compare/{data}_{variant}_{prior}.out runpygpu_chip.sh compare.py --bank banks/{data}.pt --model weights/{data}_{variant}/ema --prior {prior} --prompt_file prompts/{data}.txt --output_dir compare/{data}_{variant}_{prior}")
