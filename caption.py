'''
caption every image of a bank built by embed.py with a VLM, and store the captions and their CLIP text
embeddings in the bank (same row order as X and T):
    bank["captions"]: list of N strings
    bank["C"]: (N,E) L2-normalized CLIP text embeddings of the captions (same CLIP as the bank)
--caption_template combines the VLM caption {blip} with metadata columns stored by embed.py --metadata_columns,
e.g. "{blip}, {style} {genre} by {artist}" for WikiArt; a template without {blip} skips the VLM
usage: sbatch runpygpu_chip.sh caption.py --bank banks/celeb.pt
'''
import argparse
import string
import time
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from experiment_helpers.gpu_details import print_details
from bank import load_clip,encode_texts_clip,load_images,compose_caption

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celeb.pt")
parser.add_argument("--save_path",type=str,default="",help="defaults to overwriting --bank")
parser.add_argument("--captioner",type=str,default="Salesforce/blip-image-captioning-large")
parser.add_argument("--prompt",type=str,default="",help="optional caption prefix for conditional captioning, e.g. 'a photo of'")
parser.add_argument("--max_new_tokens",type=int,default=40)
parser.add_argument("--num_beams",type=int,default=3)
parser.add_argument("--no_repeat_ngram_size",type=int,default=3,help="block repeated n-grams (BLIP otherwise sometimes loops, e.g. 'mustache mustache mustache')")
parser.add_argument("--caption_template",type=str,default="{blip}",help="final caption; {blip} is the VLM caption, other fields are bank['metadata'] columns")
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--mixed_precision",type=str,default="fp16",choices=["no","fp16","bf16"])


def caption_images(args,bank:dict,torch_dtype,device)->list:
    embed_args=bank["args"]
    # iterate the dataset exactly as embed.py did so captions line up with the bank rows
    dataset=load_images(embed_args["dataset"],embed_args["split"],embed_args["limit"],embed_args.get("data_files"),
                        embed_args.get("shuffle",False),embed_args.get("seed",123))
    # held-out rows (embed.py --holdout) come last and aren't captioned
    dataset=dataset.select(range(bank["X"].shape[0]))

    processor=BlipProcessor.from_pretrained(args.captioner)
    captioner=BlipForConditionalGeneration.from_pretrained(args.captioner,torch_dtype=torch_dtype).to(device).eval()

    captions=[]
    for start in range(0,len(dataset),args.batch_size):
        images=[img.convert("RGB") for img in dataset[start:start+args.batch_size][embed_args["image_column"]]]
        if args.prompt:
            inputs=processor(images=images,text=[args.prompt]*len(images),return_tensors="pt").to(device)
        else:
            inputs=processor(images=images,return_tensors="pt").to(device)
        inputs["pixel_values"]=inputs["pixel_values"].to(torch_dtype)
        with torch.no_grad():
            out=captioner.generate(**inputs,max_new_tokens=args.max_new_tokens,num_beams=args.num_beams,no_repeat_ngram_size=args.no_repeat_ngram_size)
        captions+=[c.strip() for c in processor.batch_decode(out,skip_special_tokens=True)]
        if (start//args.batch_size)%20==0:
            print(f"\t captioned {len(captions)}/{len(dataset)}: {captions[-1]}")
    return captions


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    if device=="cpu":
        torch_dtype=torch.float32

    bank=torch.load(args.bank,weights_only=False)
    metadata=bank.get("metadata",{})
    n=bank["X"].shape[0]
    fields={f[1] for f in string.Formatter().parse(args.caption_template) if f[1]}
    missing=fields-{"blip"}-set(metadata)
    assert not missing,f"--caption_template uses {missing}, not in the bank's metadata columns {list(metadata)} (embed.py --metadata_columns)"
    blip=caption_images(args,bank,torch_dtype,device) if "blip" in fields else [""]*n
    captions=[compose_caption(args.caption_template,blip=blip[i],**{c:v[i] for c,v in metadata.items()}) for i in range(n)]
    assert all(captions),"some captions came out empty; check --caption_template against the metadata"

    clip_model,clip_processor=load_clip(bank["clip_model"],device,torch_dtype)
    C=torch.cat([encode_texts_clip(clip_model,clip_processor,captions[i:i+256],device).cpu()
                 for i in range(0,len(captions),256)])
    assert torch.isfinite(C).all(),"non-finite caption embeddings, try --mixed_precision no"
    bank["captions"]=captions
    bank["C"]=C
    bank["caption_args"]=vars(args)
    save_path=args.save_path or args.bank
    torch.save(bank,save_path)
    print(f"saved {len(captions)} captions and C {tuple(C.shape)} to {save_path}; examples: {captions[:3]}")


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")
