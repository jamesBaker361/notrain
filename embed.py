'''
build the bank for text-conditioned CFDM / the CLIP tree:
    X: (N,D) flattened training points: pixels in [-1,1] (--space pixel), SD-VAE latents (--space vae), or the
       VQ latents of a latent-diffusion repo such as CompVis/ldm-celebahq-256 (--space vq)
    T: (N,E) L2-normalized CLIP image embeddings of the same images
    X_test/T_test: held-out images, from a separate split (--test_split) or the last --holdout training rows;
       never in the tree or the samplers, only the FID/KID reference
    captions/C (+ captions_test): with --label_column, captions from class names via --caption_template and
       their CLIP text embeddings (otherwise run caption.py for BLIP captions)
usage: sbatch runpygpu_chip.sh embed.py --dataset uoft-cs/cifar10 --image_column img --test_split test --label_column label --space pixel --resolution 32
       sbatch runpygpu_chip.sh embed.py --dataset korexyz/celeba-hq-256x256 --test_split validation --space vq --resolution 256
'''
import os
import argparse
import time
import torch
import torchvision.transforms as transforms
from experiment_helpers.gpu_details import print_details
from bank import load_images,metadata_strings,load_clip,encode_images_clip,encode_texts_clip,load_vae,encode_vae

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="korexyz/celeba-hq-256x256")
parser.add_argument("--split",type=str,default="train")
parser.add_argument("--test_split",type=str,default="",help="held-out split stored as X_test (e.g. test, validation); overrides --holdout")
parser.add_argument("--image_column",type=str,default="image")
parser.add_argument("--label_column",type=str,default="",help="class-label column; with it, captions come from --caption_template and the class names")
parser.add_argument("--caption_template",type=str,default="a photo of a {}")
parser.add_argument("--data_files",type=str,nargs="*",default=None,help="only these shards (globs, relative to the dataset repo), e.g. data/train-0000*.parquet for part of huggan/wikiart")
parser.add_argument("--shuffle",action="store_true",help="shuffle each split (with --seed) before --limit/--test_limit; for class-sorted datasets like ImageNet")
parser.add_argument("--seed",type=int,default=123)
parser.add_argument("--metadata_columns",type=str,nargs="*",default=[],help="columns kept as strings in bank['metadata'] for caption.py --caption_template, e.g. artist genre style")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--test_limit",type=int,default=-1)
parser.add_argument("--resolution",type=int,default=256)
parser.add_argument("--space",type=str,default="vae",choices=["vae","vq","pixel"])
parser.add_argument("--vae",type=str,default="",help="autoencoder repo; defaults to stabilityai/sd-vae-ft-mse (vae) or CompVis/ldm-celebahq-256 (vq)")
parser.add_argument("--clip_model",type=str,default="openai/clip-vit-large-patch14")
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--mixed_precision",type=str,default="fp16",choices=["no","fp16","bf16"])
parser.add_argument("--store_dtype",type=str,default="float32",choices=["float32","float16"],help="float16 halves the bank, e.g. 256px pixel CelebA-HQ is 11GB instead of 22GB")
parser.add_argument("--holdout",type=int,default=0,help="without --test_split: keep the last k images out of X/T as X_test/T_test")
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
    store_dtype={"float32":torch.float32,"float16":torch.float16}[args.store_dtype]
    vae_name=args.vae or {"vae":"stabilityai/sd-vae-ft-mse","vq":"CompVis/ldm-celebahq-256"}.get(args.space)

    def load(split:str,limit:int):
        return load_images(args.dataset,split,limit,args.data_files,args.shuffle,args.seed)

    dataset=load(args.split,args.limit)
    test_dataset=load(args.test_split,args.test_limit) if args.test_split else None
    print("dataset",dataset,"test",test_dataset)
    if args.space=="pixel":
        d=3*args.resolution**2
        bytes_per=2 if args.store_dtype=="float16" else 4
        print(f"pixel bank: {len(dataset)} x {d} = {len(dataset)*d*bytes_per/1e9:.1f}GB; one covariance for the locality sampler = {d*d*4/1e9:.1f}GB")
        if args.resolution>64:
            print("warning: pixel space above 64x64 makes the locality/wiener node samplers impractical (d x d covariance)")

    transform=transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    clip_model,clip_processor=load_clip(args.clip_model,device,torch_dtype)
    vae=load_vae(vae_name,device,torch_dtype,args.space) if args.space!="pixel" else None

    def embed(ds):
        X_buffer=[]
        T_buffer=[]
        for start in range(0,len(ds),args.batch_size):
            images=[img.convert("RGB") for img in ds[start:start+args.batch_size][args.image_column]]
            pixels=torch.stack([transform(img) for img in images]).to(device)
            x=encode_vae(vae,pixels) if vae is not None else pixels.float()
            # fp16 encoders can overflow; a single nan row would poison every softmax in the sampler
            assert torch.isfinite(x).all(),"non-finite values in bank, try --mixed_precision no"
            X_buffer.append(x.flatten(1).to(store_dtype).cpu())
            T_buffer.append(encode_images_clip(clip_model,clip_processor,images,device).cpu())
            if (start//args.batch_size)%20==0:
                print(f"\t embedded {start+len(images)}/{len(ds)}")
        return torch.cat(X_buffer),torch.cat(T_buffer),tuple(x.shape[1:])

    def label_captions(ds):
        names=ds.features[args.label_column].names
        return [args.caption_template.format(names[l]) for l in ds[args.label_column]]

    def caption_embeds(captions:list)->torch.Tensor:
        # few distinct captions for class labels: encode each once
        unique=sorted(set(captions))
        emb=torch.cat([encode_texts_clip(clip_model,clip_processor,unique[i:i+256],device).cpu() for i in range(0,len(unique),256)])
        index={c:i for i,c in enumerate(unique)}
        return emb[[index[c] for c in captions]]

    X,T,shape=embed(dataset)
    metadata={c:metadata_strings(dataset,c) for c in args.metadata_columns}
    metadata_test={c:metadata_strings(test_dataset,c) for c in args.metadata_columns} if test_dataset is not None else None
    captions=label_captions(dataset) if args.label_column else None
    if test_dataset is not None:
        X_test,T_test,_=embed(test_dataset)
        captions_test=label_captions(test_dataset) if args.label_column else None
    else:
        n_train=X.shape[0]-args.holdout
        assert n_train>0,"--holdout must leave some training images"
        X,X_test,T,T_test=X[:n_train],X[n_train:],T[:n_train],T[n_train:]
        metadata_test={c:v[n_train:] for c,v in metadata.items()}
        metadata={c:v[:n_train] for c,v in metadata.items()}
        if captions is not None:
            captions,captions_test=captions[:n_train],captions[n_train:]
        else:
            captions_test=None
    bank={
        "X":X,
        "T":T,
        "X_test":X_test,
        "T_test":T_test,
        "shape":shape,
        "space":args.space,
        "vae":vae_name if args.space!="pixel" else None,
        "clip_model":args.clip_model,
        "logit_scale":clip_model.logit_scale.exp().item(),
        "args":vars(args)
    }
    if metadata:
        bank["metadata"]=metadata
        bank["metadata_test"]=metadata_test
    if captions is not None:
        bank["captions"]=captions
        bank["C"]=caption_embeds(captions)
        bank["captions_test"]=captions_test
    os.makedirs(os.path.dirname(args.save_path) or ".",exist_ok=True)
    torch.save(bank,args.save_path)
    print(f"saved X {tuple(X.shape)} {X.dtype} T {tuple(T.shape)} X_test {tuple(X_test.shape)} shape {shape} "
          f"captions {'yes, e.g. '+repr(captions[0]) if captions else 'no (run caption.py)'} to {args.save_path}")


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
