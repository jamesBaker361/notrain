# celeb uses the standard CelebA-HQ train/validation split; birds and league have no standard copy
datasets={"celeb":"korexyz/celeba-hq-256x256 --test_split validation","birds":"jlbaker361/birds","league-splash-tagged":"jlbaker361/league-splash-tagged"}
for data,dataset in datasets.items():
    print(f"sbatch -J embed --err=slurm_chip/embed/{data}.err --out=slurm_chip/embed/{data}.out runpygpu_chip.sh embed.py --dataset {dataset} --save_path banks/{data}.pt  --space vae --resolution 256")
