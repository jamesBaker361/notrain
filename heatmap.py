'''
text heatmaps over CLIP image tokens (Gandelsman et al. 2024, "Interpreting CLIP's image representation via
text-based decomposition", the clip_text_span submodule): the CLIP image embedding is, up to the final
LayerNorm, a sum of residual-stream terms, and the attention terms of the last few layers carry almost all of
it (their mean-ablation of everything else barely moves zero-shot accuracy). The attention output at the class
token is itself a sum over image tokens, so token i's direct contribution c_i, projected into the joint space,
gives the heatmap <c_i, text>: how much that region writes the text direction.

This reimplements their prs_hook on the HF CLIPModel the bank was embedded with (openai/clip-vit-large-patch14
by default, 16x16 tokens at 224px), so the heatmaps share the bank's text embeddings. The vision transformer
is run layer by layer from its submodules, which keeps this independent of how a transformers version
returns hidden states.
'''
import torch
import torch.nn.functional as F
from transformers import CLIPModel


def _heads(x:torch.Tensor,attn)->torch.Tensor:
    # (B,N,D) -> (B,H,N,head_dim)
    B,N,_=x.shape
    return x.reshape(B,N,attn.num_heads,attn.head_dim).transpose(1,2)


@torch.no_grad()
def residual_stream(model:CLIPModel,pixel_values:torch.Tensor,n_layers:int=4):
    '''
    runs the vision tower and returns (r, contribs): r (B,width) the final class-token residual (the input of
    post_layernorm), and contribs (B,n_layers,N,width), the direct contribution of each token (class token
    first) to r through each of the last n_layers attention blocks; with the out_proj bias they sum to that
    block's class-token output
    '''
    vision=model.vision_model
    h=vision.pre_layrnorm(vision.embeddings(pixel_values))
    layers=vision.encoder.layers
    contribs=[]
    for l,layer in enumerate(layers):
        attn=layer.self_attn
        x=layer.layer_norm1(h)
        q,k,v=_heads(attn.q_proj(x),attn),_heads(attn.k_proj(x),attn),_heads(attn.v_proj(x),attn)
        out=F.scaled_dot_product_attention(q,k,v,scale=attn.scale)
        if l>=len(layers)-n_layers:
            # class-token attention row, then each token's value through its head's slice of out_proj
            alpha=torch.softmax(q[:,:,:1]@k.transpose(-1,-2)*attn.scale,dim=-1)[:,:,0]
            weighted=(alpha[...,None]*v).transpose(1,2).flatten(2)
            contribs.append(weighted@attn.out_proj.weight.T)
        h=h+attn.out_proj(out.transpose(1,2).flatten(2))
        h=h+layer.mlp(layer.layer_norm2(h))
    return h[:,0],torch.stack(contribs,dim=1)


def project(model:CLIPModel,c:torch.Tensor,r:torch.Tensor)->torch.Tensor:
    '''
    a residual term c (...,width) mapped into the joint embedding space with the final LayerNorm linearized at
    the full residual r (B,width): gamma*(c - mean(c))/std(r), then the visual projection. Summing this over
    every term of r gives the image embedding minus the LayerNorm's beta (projected)
    '''
    ln=model.vision_model.post_layernorm
    std=(r.var(-1,unbiased=False)+ln.eps).sqrt()
    std=std.reshape(std.shape[0],*[1]*(c.dim()-2),1)
    return (ln.weight*(c-c.mean(-1,keepdim=True))/std)@model.visual_projection.weight.T


@torch.no_grad()
def token_heatmaps(model:CLIPModel,processor,images:list,text_embeds:torch.Tensor,device,n_layers:int=4,
                   batch_size:int=16)->torch.Tensor:
    '''
    images: list of PIL images, text_embeds: (P,E) L2-normalized -> (B,P,g,g) heatmaps on the g x g token grid,
    each image's contributions divided by its embedding norm (so a map sums to roughly its cosine similarity).
    The processor center-crops, so the grid covers the whole image only for square images.
    Subtract a mean over several texts before using these: raw maps share a text-independent component
    '''
    maps=[]
    for i in range(0,len(images),batch_size):
        pixel_values=processor(images=images[i:i+batch_size],return_tensors="pt")["pixel_values"].to(device,model.dtype)
        r,contribs=residual_stream(model,pixel_values,n_layers)
        embed=model.visual_projection(model.vision_model.post_layernorm(r))
        # image tokens only: the class token's own term has a negligible direct effect (their mean-ablation)
        tokens=project(model,contribs[:,:,1:].sum(1),r)/embed.norm(dim=-1)[:,None,None]
        g=int(round(tokens.shape[1]**0.5))
        maps.append((tokens.float()@text_embeds.T.float()).transpose(1,2).reshape(-1,text_embeds.shape[0],g,g))
    return torch.cat(maps)
