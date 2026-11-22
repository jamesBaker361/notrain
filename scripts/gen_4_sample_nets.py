# stage 4 (after 2): samples from the trained nets, with the same metrics as the closed-form runs
# usage: python scripts/gen_4_sample_nets.py [setting ...] | bash
from experiment_config import selected_settings,setting,sbatch,finished,metrics

for name in selected_settings():
    cfg=setting(name)
    for kind,spec in cfg["nets"].items():
        # the as-is checkpoint is unconditional, so only w=0 means anything; the cross-attention nets use w as the CFG scale
        ws="0" if kind=="pretrained" else "0 1 4"
        run=f"{name}_net_{kind}{cfg['sharp_suffix']}"
        if finished(f"outputs/{run}"):
            continue
        print(sbatch(f"net_{kind}",f"sample/{run}",f"sample.py --bank {cfg['bank']} --prompt_file {cfg['prompts']} --method net --model {spec} --steps 50 --w {ws} {metrics} {cfg['sharp_flags']} --output_dir outputs/{run}"))
