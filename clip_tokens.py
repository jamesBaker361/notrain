'''
store the CLIP token embeddings of every bank image (Gandelsman et al., "Interpreting CLIP's image representation
via text-based decomposition", the clip_text_span submodule; heatmap.py) next to a bank built by embed.py:
    tokens: (N,g,g,E) float16, same row order as X and T. Token (i,j) is the direct contribution of the (i,j)
            cell of the image (16x16 cells for ViT-L/14) to its CLIP embedding, through the last --n_layers
            attention blocks, in the joint space where the text embeddings live; tokens @ text is the heatmap
the images are re-read from the dataset with the bank's embed.py args (as caption.py does) and seen exactly as
for T: CLIP's center crop of the original image, which is also the crop embed.py resized into X. So token
(i,j) covers the (i,j) cell of X at any resolution, and pyramid.py maps it to the denoiser patches overlapping
it at each level. ~0.4MB per image (11GB for 28k), saved to <bank>_tokens.pt and memory-mapped by sample.py
usage: sbatch runpygpu_chip.sh clip_tokens.py --bank banks/celebahq_pix64.pt
'''
import argparse
import time
import torch
from experiment_helpers.gpu_details import print_details
from bank import load_clip,bank_dataset,tokens_path
from heatmap import token_embeddings

parser=argparse.ArgumentParser()
parser.add_argument("--bank",type=str,default="banks/celebahq_pix64.pt")
parser.add_argument("--save_path",type=str,default="",help="defaults to <bank>_tokens.pt")
parser.add_argument("--n_layers",type=int,default=4,help="last attention layers whose direct effects are summed")
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--mixed_precision",type=str,default="fp16",choices=["no","fp16","bf16"])


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    if device=="cpu":
        torch_dtype=torch.float32
    # memory-mapped: only the args and row count are needed, not the (possibly 11GB) X
    bank=torch.load(args.bank,weights_only=False,mmap=True)
    dataset=bank_dataset(bank)
    image_column=bank["args"]["image_column"]
    clip_model,clip_processor=load_clip(bank["clip_model"],device,torch_dtype)
    tokens=None
    for start in range(0,len(dataset),args.batch_size):
        images=[img.convert("RGB") for img in dataset[start:start+args.batch_size][image_column]]
        batch=token_embeddings(clip_model,clip_processor,images,device,args.n_layers,args.batch_size)
        assert torch.isfinite(batch).all(),"non-finite token embeddings, try --mixed_precision no"
        if tokens is None:
            # filled in place: a list of chunks would need twice the memory at the final cat
            tokens=torch.empty((len(dataset),*batch.shape[1:]),dtype=torch.float16)
        tokens[start:start+len(images)]=batch.half().cpu()
        if (start//args.batch_size)%20==0:
            print(f"\t tokens {start+len(images)}/{len(dataset)}")
    save_path=args.save_path or tokens_path(args.bank)
    torch.save({"tokens":tokens,"clip_model":bank["clip_model"],"n_layers":args.n_layers,"args":vars(args)},save_path)
    print(f"saved tokens {tuple(tokens.shape)} ({tokens.numel()*2/1e9:.1f}GB) to {save_path}")


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
