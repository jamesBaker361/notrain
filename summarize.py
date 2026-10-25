'''
collect sample.py runs into one comparison: a table of metrics averaged over prompts for each
(node sampler, tree depth, w), and per (prompt, w) an image stacking every run's samples with a label
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

METRICS=["clip_score","copy_rate","nn_dist","node_size","distinct_nodes","seconds_per_sample"]


def run_label(args:dict)->str:
    if args.get("method")=="tree":
        return f"{args['node_sampler']} L{args['tree_depth']}"
    return f"{args.get('method','cfdm')} scales{args.get('scales')}"


def main(args):
    run_dirs=sorted({d for pattern in args.runs for d in glob.glob(pattern) if os.path.isfile(os.path.join(d,"metrics.json"))})
    assert run_dirs,f"no runs with metrics.json in {args.runs}"
    runs=[]
    for d in run_dirs:
        with open(os.path.join(d,"metrics.json")) as f:
            m=json.load(f)
        runs.append({"dir":d,"args":m["args"],"prompts":m["prompts"],"results":m["results"],"label":run_label(m["args"])})
    # order rows by sampler, then depth
    runs.sort(key=lambda r:(r["args"].get("node_sampler",""),r["args"].get("tree_depth",0),r["dir"]))
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
    with open(os.path.join(args.output_dir,"summary.csv"),"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    header="| run | w | "+" | ".join(METRICS)+" |"
    lines=[header,"|"+"---|"*(len(METRICS)+2)]
    for e in table:
        lines.append(f"| {e['run']} | {e['w']} | "+" | ".join("-" if e[k] is None else f"{e[k]:.3f}" for k in METRICS)+" |")
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
