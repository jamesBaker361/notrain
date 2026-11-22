# stage 2 (after 1): the text-conditional nets, fine-tuned from the checkpoint ("ft") and from scratch ("scratch")
# usage: python scripts/gen_2_train.py [setting ...] | bash
from experiment_config import selected_settings,setting,sbatch,train_flags

for name in selected_settings():
    cfg=setting(name)
    for kind,flags in train_flags.items():
        if kind not in cfg["nets"]:
            continue
        print(sbatch(f"{kind}_{name}",f"train/{name}_{kind}",
                     f"finetune.py --bank {cfg['bank']} --base {cfg['base']} --name {name}_{kind} {flags} {cfg['size']} {cfg['train']}",
                     script=cfg.get("train_script","runpygpu_chip.sh")))
