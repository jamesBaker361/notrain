'''
evaluation metrics for generated samples (images as (B,3,H,W) in [-1,1])

ReferenceMetrics, against a fixed reference set (needs torch-fidelity):
    FID: Frechet distance between Inception features; biased upward at small sample sizes, so only compare
         runs with the same number of samples
    KID: unbiased kernel version, usable at a few hundred samples (mean and std over subsets)
    precision / recall (Kynkaanniemi et al. 2019): fraction of samples inside the k-NN balls of the reference
         Inception features (quality), and fraction of reference images inside the samples' balls (coverage)
    reference features are computed once and kept across resets, so one object scores every (run, w).
    measure against held-out images: against the training set, a sampler that copies training images looks
    perfect, so "vs train" is only useful next to "vs test" (a much better train score means copying)
DinoCopy: nearest-training-image cosine similarity in DINOv2 feature space, which catches near-copies
    (shifted, recolored, blended) that a pixel distance misses
LpipsDiversity: mean pairwise LPIPS between samples of the same prompt (perceptual diversity)
prompt_accuracy: fraction of samples whose best-matching prompt (by an independent CLIP) is their own
'''
import os
import random
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def to_uint8(pixels:torch.Tensor)->torch.Tensor:
    # (B,3,H,W) in [-1,1] -> uint8
    return ((pixels.clamp(-1,1)+1)*127.5).round().to(torch.uint8)


def knn_radii(feats:torch.Tensor,k:int)->torch.Tensor:
    # distance from each point to its k-th nearest neighbor in the same set (excluding itself)
    d=torch.cdist(feats,feats)
    return d.kthvalue(k+1,dim=1).values


def precision_recall(real:torch.Tensor,fake:torch.Tensor,k:int=3)->tuple:
    '''
    improved precision/recall: a sample is "precise" if it falls in some real point's k-NN ball, and a real
    point is "recalled" if it falls in some sample's k-NN ball
    '''
    real,fake=real.double(),fake.double()
    r_real=knn_radii(real,k)
    r_fake=knn_radii(fake,k)
    d=torch.cdist(fake,real)
    precision=(d<=r_real[None]).any(1).float().mean().item()
    recall=(d.T<=r_fake[None]).any(1).float().mean().item()
    return precision,recall


class ReferenceMetrics:
    def __init__(self,reference_pixels:torch.Tensor,device,kid_subset:int=100,batch_size:int=64,pr_k:int=3):
        '''
        reference_pixels: (N,3,H,W) in [-1,1]; kid_subset must be <= the number of generated samples
        '''
        self.device=device
        self.batch_size=batch_size
        self.pr_k=pr_k
        self.n_ref=reference_pixels.shape[0]
        self.fid=FrechetInceptionDistance(feature=2048,reset_real_features=False).to(device)
        self.kid=KernelInceptionDistance(subset_size=min(kid_subset,self.n_ref),reset_real_features=False).to(device)
        real=[]
        for i in range(0,self.n_ref,batch_size):
            batch=to_uint8(reference_pixels[i:i+batch_size]).to(device)
            self.fid.update(batch,real=True)
            self.kid.update(batch,real=True)
            real.append(self.features(batch))
        self.real_feats=torch.cat(real)
        self.fake_feats=[]

    @torch.no_grad()
    def features(self,batch_uint8:torch.Tensor)->torch.Tensor:
        # the same 2048-d Inception pool features FID uses
        return self.fid.inception(batch_uint8).reshape(batch_uint8.shape[0],-1).float()

    def reset(self):
        self.fid.reset()
        self.kid.reset()
        self.fake_feats=[]

    def update(self,pixels:torch.Tensor):
        for i in range(0,pixels.shape[0],self.batch_size):
            batch=to_uint8(pixels[i:i+self.batch_size]).to(self.device)
            self.fid.update(batch,real=False)
            self.kid.update(batch,real=False)
            self.fake_feats.append(self.features(batch))

    def compute(self)->dict:
        kid_mean,kid_std=self.kid.compute()
        precision,recall=precision_recall(self.real_feats,torch.cat(self.fake_feats),self.pr_k)
        return {"fid":float(self.fid.compute()),"kid":float(kid_mean),"kid_std":float(kid_std),
                "precision":precision,"recall":recall}


class DinoCopy:
    '''
    copy detection in DINOv2 feature space. Training-image features are cached at cache_path so the many
    sample.py jobs over one bank only embed the training set once
    '''
    def __init__(self,model_name:str,device,batch_size:int=64):
        from transformers import AutoModel
        self.device=device
        self.batch_size=batch_size
        self.model=AutoModel.from_pretrained(model_name).to(device).eval()
        self.model_name=model_name
        # ImageNet normalization at DINOv2's 224 input size
        self.mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
        self.std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
        self.train_feats=None

    @torch.no_grad()
    def embed(self,pixels:torch.Tensor)->torch.Tensor:
        out=[]
        for i in range(0,pixels.shape[0],self.batch_size):
            x=(pixels[i:i+self.batch_size].to(self.device).float().clamp(-1,1)+1)/2
            x=F.interpolate(x,size=224,mode="bicubic",align_corners=False)
            x=(x-self.mean)/self.std
            cls=self.model(pixel_values=x).last_hidden_state[:,0]
            out.append(F.normalize(cls.float(),dim=-1))
        return torch.cat(out)

    def set_train(self,pixel_fn,n:int,cache_path:str=""):
        '''
        pixel_fn(i, j) -> training images i..j as pixels; features cached at cache_path when given
        '''
        if cache_path and os.path.exists(cache_path):
            cached=torch.load(cache_path,weights_only=False)
            if cached["model"]==self.model_name and cached["feats"].shape[0]==n:
                self.train_feats=cached["feats"].to(self.device)
                return
        self.train_feats=torch.cat([self.embed(pixel_fn(i,min(i+256,n))) for i in range(0,n,256)])
        if cache_path:
            tmp=cache_path+f".tmp{os.getpid()}"
            torch.save({"model":self.model_name,"feats":self.train_feats.cpu()},tmp)
            os.replace(tmp,cache_path)

    def nearest_similarity(self,pixels:torch.Tensor)->torch.Tensor:
        feats=self.embed(pixels)
        return torch.cat([(feats[i:i+256]@self.train_feats.T).amax(-1) for i in range(0,feats.shape[0],256)])


class LpipsDiversity:
    def __init__(self,device,net:str="alex"):
        import lpips
        self.device=device
        self.model=lpips.LPIPS(net=net,verbose=False).to(device).eval()

    @torch.no_grad()
    def mean_pairwise(self,pixels:torch.Tensor,max_pairs:int=50,seed:int=0)->float:
        n=pixels.shape[0]
        pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
        if not pairs:
            return None
        random.Random(seed).shuffle(pairs)
        pairs=pairs[:max_pairs]
        a=pixels[[i for i,_ in pairs]].to(self.device).float()
        b=pixels[[j for _,j in pairs]].to(self.device).float()
        return float(self.model(a,b).mean())


def prompt_accuracy(image_embeds:torch.Tensor,text_embeds:torch.Tensor,prompt_ids:torch.Tensor)->float:
    '''
    image_embeds (B,E), text_embeds (P,E), both normalized; prompt_ids (B,) the prompt each sample was made for
    '''
    return float(((image_embeds@text_embeds.T).argmax(-1)==prompt_ids.to(image_embeds.device)).float().mean())
