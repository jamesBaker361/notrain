# stage 1: embed -> (BLIP captions) -> CLIP token embeddings (clip_tokens.py, for the pyramid's heatmaps): only the
# next missing step is submitted (caption.py rewrites the bank, so nothing else may read it meanwhile); re-run once
# those jobs are done to submit the step after, until every setting reports "already built"
# (jobs still queued or running aren't detected: don't re-run until they're done)
# usage: python scripts/gen_1_build.py [setting ...] | bash
import os
import sys
from experiment_config import selected_settings,setting,has_captions,sbatch

# shared token files already submitted by an earlier setting in this run
tokens_queued=set()
for name in selected_settings():
    cfg=setting(name)
    bank=cfg["bank"]
    if not os.path.exists(bank):
        print(sbatch("embed_"+name,"embed/"+name,f"embed.py {cfg['embed']} --save_path {bank}"))
        continue
    captioned=has_captions(bank)
    if captioned is None:
        print(f"# {name}: {bank} can't be read (still being written?), skipping it",file=sys.stderr)
    elif cfg["blip"] and not captioned:
        print(sbatch("caption_"+name,"caption/"+name,f"caption.py --bank {bank} {cfg.get('caption','')}"))
    elif not os.path.exists(cfg["tokens"]) and cfg["tokens"] not in tokens_queued:
        # one shared token file per image set, from whichever of its banks is ready first
        tokens_queued.add(cfg["tokens"])
        log=os.path.basename(cfg["tokens"])[:-3]
        print(sbatch("tokens_"+name,"tokens/"+log,f"clip_tokens.py --bank {bank} --save_path {cfg['tokens']}"))
    elif os.path.exists(cfg["tokens"]):
        print(f"# {name}: bank, captions and CLIP tokens already built",file=sys.stderr)
