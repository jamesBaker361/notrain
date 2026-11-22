# stage 6 (after 1; can run alongside 2-5): exact attribution of the pyramid and whole-image CFDM samples, scored
# by leave-k-out counterfactuals against DINO / CLIP / pixel nearest-neighbour and random rankings (attribution.py)
# usage: python scripts/gen_6_attribution.py [setting ...] | bash
from experiment_config import selected_settings,setting,sbatch,finished,cfdm_flags

for name in selected_settings():
    cfg=setting(name)
    runs={"pyramid":f"--method pyramid {cfdm_flags} --pyramid_steps 10 {cfg.get('pyramid','')} --tokens {cfg['tokens']}"}
    if cfg.get("cfdm",True):
        runs["cfdm"]=f"--method cfdm {cfdm_flags}"
    for label,flags in runs.items():
        run=f"attribution_{name}_{label}"
        if finished(f"outputs/{run}","attribution.json"):
            continue
        print(sbatch(f"attr_{label}",f"attribution/{name}_{label}",f"attribution.py --bank {cfg['bank']} --prompt_file {cfg['prompts']} --n_per_prompt 8 {flags} --output_dir outputs/{run}"))
