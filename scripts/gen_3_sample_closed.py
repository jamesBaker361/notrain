# stage 3 (after 1; can run alongside 2): the closed-form samplers: the retrieval baseline, Kamb & Ganguli's ls
# machine, the whole-image CFDM on the prompt's top-K CLIP matches, the pyramid's CFDM layout alone (upsampled:
# what the patch levels add), and each pyramid variant (pyramid_variants)
# usage: python scripts/gen_3_sample_closed.py [setting ...] | bash
from experiment_config import selected_settings,setting,sbatch,finished,metrics,cfdm_flags,pyramid_variants

for name in selected_settings():
    cfg=setting(name)
    # the baseline: training images drawn from the prompt's top matches (what sigma -> 0 collapses to)
    runs={"retrieval":"--method retrieval --top_k 1000"}
    if cfg.get("ls",True):
        runs["ls"]="--method ls --top_k 1000 --steps 10"
    if cfg.get("cfdm",True):
        runs["cfdm"]=f"--method cfdm {cfdm_flags}"
    runs["cfdm_upsample"]=f"--method pyramid --pyramid_upsample {cfdm_flags} {cfg.get('pyramid','')}"
    for variant,flags in pyramid_variants.items():
        runs[variant]=f"--method pyramid {cfdm_flags} --pyramid_steps 10 {cfg.get('pyramid','')} --tokens {cfg['tokens']} {flags}"
    for label,flags in runs.items():
        run=f"{name}_{label}{cfg['sharp_suffix']}"
        if finished(f"outputs/{run}"):
            continue
        print(sbatch(label,f"sample/{run}",f"sample.py --bank {cfg['bank']} --prompt_file {cfg['prompts']} --w 0 1 4 {metrics} {flags} {cfg['sharp_flags']} --output_dir outputs/{run}"))
