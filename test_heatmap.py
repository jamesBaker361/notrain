import torch
import pytest
from transformers import CLIPConfig,CLIPModel,CLIPImageProcessor
from heatmap import residual_stream,project,token_heatmaps
from bank import to_pil


@pytest.fixture
def tiny_clip():
    torch.manual_seed(0)
    config=CLIPConfig(
        text_config={"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":4,"vocab_size":100},
        vision_config={"hidden_size":32,"intermediate_size":64,"num_hidden_layers":3,"num_attention_heads":4,
                       "image_size":32,"patch_size":8},
        projection_dim=16)
    model=CLIPModel(config).eval().double()
    pixel_values=torch.randn(2,3,32,32,dtype=torch.float64)
    return model,pixel_values


def test_manual_forward_matches_image_features(tiny_clip):
    model,pixel_values=tiny_clip
    r,_=residual_stream(model,pixel_values,n_layers=1)
    embed=model.visual_projection(model.vision_model.post_layernorm(r))
    out=model.get_image_features(pixel_values=pixel_values)
    expected=out if isinstance(out,torch.Tensor) else out.pooler_output
    assert torch.allclose(embed,expected,atol=1e-10)


def test_token_terms_sum_to_attention_output(tiny_clip):
    model,pixel_values=tiny_clip
    layers=model.vision_model.encoder.layers
    captured={}
    hooks=[layer.self_attn.register_forward_hook(lambda m,i,o,l=l:captured.__setitem__(l,o[0] if isinstance(o,tuple) else o))
           for l,layer in enumerate(layers)]
    with torch.no_grad():
        model.get_image_features(pixel_values=pixel_values)
    for h in hooks:
        h.remove()
    _,contribs=residual_stream(model,pixel_values,n_layers=len(layers))
    for l,layer in enumerate(layers):
        assert torch.allclose(contribs[:,l].sum(1)+layer.self_attn.out_proj.bias,captured[l][:,0],atol=1e-10)


def test_projection_linearizes_layernorm(tiny_clip):
    model,pixel_values=tiny_clip
    r,_=residual_stream(model,pixel_values)
    ln=model.vision_model.post_layernorm
    beta=ln.bias@model.visual_projection.weight.T
    assert torch.allclose(project(model,r,r)+beta,model.visual_projection(ln(r)),atol=1e-10)
    # linear in the term: two halves project to the halves of the whole
    assert torch.allclose(project(model,r/2,r)*2,project(model,r,r),atol=1e-12)


def test_heatmap_shape(tiny_clip):
    model,pixel_values=tiny_clip
    processor=CLIPImageProcessor(size={"shortest_edge":32},crop_size={"height":32,"width":32})
    images=to_pil(torch.rand(3,3,32,32)*2-1)
    texts=torch.nn.functional.normalize(torch.randn(5,16,dtype=torch.float64),dim=-1)
    maps=token_heatmaps(model,processor,images,texts,"cpu",n_layers=2,batch_size=2)
    assert maps.shape==(3,5,4,4)
    assert torch.isfinite(maps).all()
