# stage 5 (after 2): how well the closed-form machines predict the net trained from scratch on exactly our bank,
# from the same noise; the checkpoint and its fine-tune serve as "another DDPM"
# usage: python scripts/gen_5_agreement.py [setting ...] | bash
from experiment_config import selected_settings,setting,sbatch,finished

for name in selected_settings():
    cfg=setting(name)
    if finished(f"outputs/agreement_{name}","agreement.json"):
        continue
    nets=cfg["nets"]
    others=" ".join(v for k,v in nets.items() if k!="scratch")
    print(sbatch(f"agree_{name}",f"agreement/{name}",f"agreement.py --bank {cfg['bank']} --model {nets['scratch']} --other_models {others} --methods ideal cfdm ls --n 64 --output_dir outputs/agreement_{name}"))
