sbatch runpygpu_chip.sh sample.py --bank banks/celeb.pt --prompt_file prompts/faces.txt --w 0 1 4 --output_dir outputs/celeb

sbatch runpygpu_chip.sh sample.py --bank banks/league-splash-tagged.pt --prompt_file prompts/league.txt --w 0 1 4 --output_dir outputs/league

sbatch runpygpu_chip.sh sample.py --bank banks/birds.pt --prompt_file prompts/birds.txt --w 0 1 4 --output_dir outputs/bird
