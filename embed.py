'''
build the bank for text-conditioned CFDM:
    X: (N,D) flattened training points, either pixels in [-1,1] or scaled VAE latents
    T: (N,E) L2-normalized CLIP image embeddings of the same images
usage: sbatch runpygpu_chip.sh embed.py --dataset jlbaker361/celeb --space vae --resolution 256
'''
import os
import argparse
import time
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from experiment_helpers.gpu_details import print_details
from bank import load_clip,encode_images_clip,load_vae,encode_vae

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/celeb")
parser.add_argument("--split",type=str,default="train")
parser.add_argument("--image_column",type=str,default="image")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--resolution",type=int,default=256)
parser.add_argument("--space",type=str,default="vae",choices=["vae","pixel"])
parser.add_argument("--vae",type=str,default="stabilityai/sd-vae-ft-mse")
parser.add_argument("--clip_model",type=str,default="openai/clip-vit-large-patch14")
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--mixed_precision",type=str,default="fp16",choices=["no","fp16","bf16"])
parser.add_argument("--save_path",type=str,default="banks/celeb_vae_256.pt")


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    if device=="cpu":
        torch_dtype=torch.float32

    dataset=load_dataset(args.dataset,split=args.split)
    if args.limit>0:
        dataset=dataset.select(range(min(args.limit,len(dataset))))
    print("dataset",dataset)
    if args.space=="pixel":
        d=3*args.resolution**2
        print(f"pixel bank: {len(dataset)} x {d} floats = {len(dataset)*d*4/1e9:.1f}GB; one covariance for the locality sampler = {d*d*4/1e9:.1f}GB")
        if args.resolution>64:
            print("warning: pixel space above 64x64 makes the locality/wiener node samplers impractical (d x d covariance); the paper uses 64x64 for CelebA-HQ")

    transform=transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    clip_model,clip_processor=load_clip(args.clip_model,device,torch_dtype)
    vae=load_vae(args.vae,device,torch_dtype) if args.space=="vae" else None

    X_buffer=[]
    T_buffer=[]
    for start in range(0,len(dataset),args.batch_size):
        images=[img.convert("RGB") for img in dataset[start:start+args.batch_size][args.image_column]]
        pixels=torch.stack([transform(img) for img in images]).to(device)
        if args.space=="vae":
            x=encode_vae(vae,pixels)
        else:
            x=pixels.float()
        X_buffer.append(x.cpu())
        T_buffer.append(encode_images_clip(clip_model,clip_processor,images,device).cpu())
        if (start//args.batch_size)%20==0:
            print(f"\t embedded {start+len(images)}/{len(dataset)}")

    X=torch.cat(X_buffer)
    T=torch.cat(T_buffer)
    # fp16 encoders can overflow; a single nan row would poison every softmax in the sampler
    assert torch.isfinite(X).all() and torch.isfinite(T).all(),"non-finite values in bank, try --mixed_precision no"
    bank={
        "X":X.flatten(1),
        "T":T,
        "shape":tuple(X.shape[1:]),
        "space":args.space,
        "vae":args.vae if args.space=="vae" else None,
        "clip_model":args.clip_model,
        "logit_scale":clip_model.logit_scale.exp().item(),
        "args":vars(args)
    }
    os.makedirs(os.path.dirname(args.save_path) or ".",exist_ok=True)
    torch.save(bank,args.save_path)
    print(f"saved X {tuple(bank['X'].shape)} T {tuple(bank['T'].shape)} shape {bank['shape']} to {args.save_path}")


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
