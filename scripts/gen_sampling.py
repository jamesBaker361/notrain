for start_t in [0.05,0.1,0.2]:
    for schedule in ["coarse_to_fine","constant"]:
        # constant needs one weight per scale
        scale_weights="--scale_weights 1 1" if schedule=="constant" else ""
        for data in ["birds","celeb","league-splash-tagged"]:
            command=f"sbatch runpygpu_chip.sh sample.py \
                --bank banks/{data}.pt --prompt_file prompts/{data}.txt \
                --w 0 1 4 --output_dir outputs/{data}_{schedule}_{start_t}\
                    --start_t {start_t} --schedule {schedule} --n_per_prompt 4 \
                        --scales 2 4 {scale_weights} "
            print(command)