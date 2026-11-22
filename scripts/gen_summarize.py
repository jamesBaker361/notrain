# prints a summarize.py command per setting of gen_experiments.py with finished runs (a metrics.json in
# outputs/<setting>_*), collecting them into outputs/summary_<setting>
# usage: python scripts/gen_summarize.py [setting ...]
import glob
import os
import sys

# the setting names of gen_experiments.py's SETTINGS
SETTINGS=["celebahq_pix64","celebahq_pix256","celebahq_ldm","celebahq_flux2"]
settings=sys.argv[1:] or SETTINGS
if any(s not in SETTINGS for s in settings):
    sys.exit(f"usage: python scripts/gen_summarize.py [setting ...]\n  settings: {' '.join(SETTINGS)}\n  got {settings}")

for name in settings:
    if not glob.glob(f"outputs/{name}_*/metrics.json"):
        print(f"# {name}: no finished runs yet",file=sys.stderr)
        continue
    print(f"python summarize.py --runs 'outputs/{name}_*' --output_dir outputs/summary_{name}")
