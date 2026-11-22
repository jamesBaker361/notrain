'''
collect sample.py runs into one comparison: a table of metrics averaged over prompts for each
(run, w), and per (prompt, w) an image stacking every run's samples with a label
usage: python summarize.py --runs 'outputs/pix_celeb_*' --output_dir outputs/summary_pix_celeb
'''
import os
import re
import csv
import glob
import json
import argparse
import numpy as np
from PIL import Image,ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument("--runs",type=str,nargs="+",required=True,help="run directories or globs, each with a metrics.json from sample.py")
parser.add_argument("--output_dir",type=str,default="outputs/summary")
parser.add_argument("--label_width",type=int,default=150)

# per-prompt metrics are averaged over prompts; fid_*/kid_* come from sample.py's --fid_samples pass ("__all__" rows)
METRICS=["clip_score","prompt_accuracy","itm_score","copy_rate","nn_dist","diversity","lpips_diversity","patches_kept","seconds_per_sample",
         "fid_test","fid_std_test","kid_test","precision_test","recall_test","fid_train","kid_train","precision_train","recall_train",
         "fid_real","fid_std_real","kid_real","precision_real","recall_real",
         "fid_copy_rate","fid_nn_dist","dino_nn_sim","dino_copy_rate","fid_prompt_accuracy","fid_itm_score","fid_lpips_diversity","fid_seconds_per_sample"]


def run_label(args:dict)->str:
    if args.get("method")=="net":
        model=args.get("model","")
        # † public checkpoints (and fine-tunes of them) were trained on all of their dataset, possibly including
        # our held-out split, so their fid_test/fid_real may be optimistic
        if model.startswith("pretrained:"):
            return f"net {model.split(':',1)[1].split('/')[-1]} (as-is) †"
        # finetune.py runs live in weights/<name>/ema
        name=os.path.basename(os.path.dirname(model)) or model
        return f"net {name}"+(" †" if name.endswith("_ft") else "")
    if args.get("method")=="pyramid" and args.get("pyramid_upsample"):
        return f"cfdm c{args['coarse_size']} upsampled (no patch levels)"
    if args.get("method")=="pyramid":
        label=f"pyramid c{args['coarse_size']} p{args['patch_size']}"
        if args.get("heatmap_keep",1.0)<1:
            label+=f" keep{args['heatmap_keep']}"+(f"+rand{args['heatmap_random']}" if args.get("heatmap_random") else "")
            if args.get("heatmap_overlap"):
                label+=f" ov{args['heatmap_overlap']}"
        if args.get("heatmap_lambda"):
            label+=f" hm{args['heatmap_lambda']}"
        return label
    if args.get("method","cfdm")=="cfdm":
        return f"cfdm K{args['top_k']}"
    if args.get("method")=="retrieval":
        return f"retrieval K{args['top_k']}"
    if args.get("method")=="ls":
        return f"ls machine K{args['top_k']}"
    # runs from removed samplers (CLIP tree, top-K node samplers) are still listed, by method and run dir
    return f"{args['method']} [{os.path.basename(os.path.normpath(args.get('output_dir','')))}]"


def main(args):
    run_dirs=sorted({d for pattern in args.runs for d in glob.glob(pattern) if os.path.isfile(os.path.join(d,"metrics.json"))})
    assert run_dirs,f"no runs with metrics.json in {args.runs}"
    runs=[]
    for d in run_dirs:
        with open(os.path.join(d,"metrics.json")) as f:
            m=json.load(f)
        runs.append({"dir":d,"args":m["args"],"prompts":m["prompts"],"results":m["results"],"label":run_label(m["args"]),
                     "ceiling":m.get("ceiling")})
    # same settings on different banks (e.g. pixel vs latent) get the same label; tell them apart by run dir
    labels=[r["label"] for r in runs]
    for r in runs:
        if labels.count(r["label"])>1:
            r["label"]+=f" [{os.path.basename(os.path.normpath(r['dir']))}]"
    # order rows by method, then label
    runs.sort(key=lambda r:(r["args"].get("method","cfdm"),r["label"],r["dir"]))
    os.makedirs(args.output_dir,exist_ok=True)

    table=[]
    for r in runs:
        for w in r["args"]["w"]:
            rows=[x for x in r["results"] if x["w"]==w]
            entry={"run":r["label"],"dir":r["dir"],"w":w}
            for k in METRICS:
                vals=[x[k] for x in rows if x.get(k) is not None]
                entry[k]=float(np.mean(vals)) if vals else None
            table.append(entry)
    # one FID ceiling row per bank (decoded training images scored like samples), from the first run that has it
    ceilings={}
    for r in runs:
        if r.get("ceiling") and r["args"]["bank"] not in ceilings:
            ceilings[r["args"]["bank"]]=r["ceiling"]
    for b,c in ceilings.items():
        table.append({"run":f"ceiling [{os.path.basename(b)}]","dir":"","w":"-",**{k:c.get(k) for k in METRICS}})
    # drop columns no run has (e.g. FID when --fid_samples wasn't used)
    columns=[k for k in METRICS if any(e[k] is not None for e in table)]
    table=[{k:v for k,v in e.items() if k in ["run","dir","w"]+columns} for e in table]
    with open(os.path.join(args.output_dir,"summary.csv"),"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    header="| run | w | "+" | ".join(columns)+" |"
    lines=[header,"|"+"---|"*(len(columns)+2)]
    for e in table:
        lines.append(f"| {e['run']} | {e['w']} | "+" | ".join("-" if e[k] is None else f"{e[k]:.3f}" for k in columns)+" |")
    if any("†" in e["run"] for e in table):
        lines+=["","† public checkpoint (or a fine-tune of one) trained on all of its dataset, possibly including the held-out split: its held-out scores may be optimistic"]
    with open(os.path.join(args.output_dir,"summary.md"),"w") as f:
        f.write("\n".join(lines)+"\n")
    print("\n".join(lines))

    # comparison images: for each prompt and w, one row per run (rows of each run's grid are its w values)
    prompts=runs[0]["prompts"]
    for p,prompt in enumerate(prompts):
        name=re.sub(r"[^A-Za-z0-9]+","_",prompt).strip("_")[:60]
        for w in runs[0]["args"]["w"]:
            strips=[]
            for r in runs:
                path=os.path.join(r["dir"],f"{p}_{name}.png")
                if not os.path.exists(path) or w not in r["args"]["w"] or r["prompts"]!=prompts:
                    continue
                grid=Image.open(path).convert("RGB")
                n_rows=len(r["args"]["w"])
                h=grid.height//n_rows
                j=r["args"]["w"].index(w)
                row=grid.crop((0,j*h,grid.width,(j+1)*h))
                strip=Image.new("RGB",(args.label_width+row.width,h),"white")
                ImageDraw.Draw(strip).text((4,h//2-6),r["label"],fill="black")
                strip.paste(row,(args.label_width,0))
                strips.append(strip)
            if not strips:
                continue
            out=Image.new("RGB",(max(s.width for s in strips),sum(s.height for s in strips)),"white")
            y=0
            for s in strips:
                out.paste(s,(0,y))
                y+=s.height
            out.save(os.path.join(args.output_dir,f"{p}_{name}_w{w}.png"))
    print(f"saved summary.csv, summary.md and per-prompt comparison images to {args.output_dir}")


if __name__=='__main__':
    main(parser.parse_args())
