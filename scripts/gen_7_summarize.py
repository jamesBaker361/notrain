# stage 7 (after 3-4, 6): a summarize.py command per setting with finished runs (a metrics.json in
# outputs/<setting>_*), collecting them into outputs/summary_<setting>; these run directly, not through sbatch
# usage: python scripts/gen_7_summarize.py [setting ...] | bash
import glob
import sys
from experiment_config import selected_settings

for name in selected_settings():
    if not glob.glob(f"outputs/{name}_*/metrics.json"):
        print(f"# {name}: no finished runs yet",file=sys.stderr)
        continue
    print(f"python summarize.py --runs 'outputs/{name}_*' --output_dir outputs/summary_{name}")
