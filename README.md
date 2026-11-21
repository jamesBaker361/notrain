# notrain

Training-free, text-conditioned image generation with smoothed closed-form diffusion models
([Scarvelis et al. 2023](https://arxiv.org/abs/2310.12395)), conditioned with CLIP.

## Idea

The closed-form model is a Gaussian mixture over training points, $\rho_t(z) = \frac1N\sum_i \mathcal N(z; t x_i, (1-t)^2 I)$.
Conditioning on text $c$ reweights the components by $p(x_i \mid c)$, which CLIP gives directly over the training set:

$$\log p(x_i\mid c) = s\cos(c,\tau_i) + \text{const} = -\tfrac{s}{2}\lVert c-\tau_i\rVert^2 + \text{const}$$

where $\tau_i$ is the CLIP image embedding of $x_i$ and $s$ is CLIP's logit scale. Because the prior doesn't depend on $z$, the conditional score stays in closed form:

$$k_t(z,c) = \sum_i \mathrm{softmax}_i\!\Big(-\tfrac{\lVert z-tx_i\rVert^2}{2(1-t)^2} + w\log p(x_i\mid c)\Big)\,t x_i,\qquad \nabla\log\rho_t(z\mid c)=\frac{k_t(z,c)-z}{(1-t)^2}$$

The guidance weight $w$ sets how much the text matters: $w=0$ is unconditional, $w=1$ is the exact conditional model, and $w>1$ sharpens it like CFG. No classifier gradients and no noisy-input classifier are needed. Smoothing (Eq. 3), the exact start at $t=T$, and Euler sampling are as in the paper.

Since $c$ is fixed for a whole trajectory, the text prior is computed once per prompt. Sampling then runs over only the top-k CLIP matches (`--top_k`). This takes the place of the paper's nearest-neighbour score estimator. The `w=0` baseline always uses the full bank, since the top-k subset would itself be text conditioning. `prior_mass` in `metrics.json` reports how much of the $w$-sharpened prior the subset keeps, and a warning prints when it's below 0.9; if that happens, raise `--top_k`.

## Files

- `cfdm.py`: the sampler (pure torch).
- `pyramid.py`: the coarse-to-fine sampler: CFDM at the coarsest level, then closed-form patch levels (see below).
- `heatmap.py`: per-token CLIP image embeddings (the text-span decomposition); `clip_tokens.py` stores them for every bank image, and their text heatmaps weight and filter the pyramid's patches.
- `bank.py`: CLIP and VAE encode/decode helpers.
- `embed.py`: builds a bank: `X` (flattened pixels or SD-VAE latents) and `T` (normalized CLIP image embeddings).
- `sample.py`: samples for each prompt in `--prompt_file` (one per line; `--prompts` overrides it) and a sweep over `w`, with `--method cfdm`, `pyramid` or `net`. It saves image grids, where each row is the top CLIP-retrieved training image followed by the samples, and writes `metrics.json` with CLIP score, copy rate and nearest-neighbour distance.
- `test_cfdm.py`: checks the score against autograd of the conditional mixture, plus the limits in `w`, top-k and sampling. `test_pyramid.py` and `test_heatmap.py` cover the other two samplers' pieces.

## Usage

```bash
git submodule update --init
sbatch runpygpu_chip.sh embed.py --dataset korexyz/celeba-hq-256x256 --test_split validation --space vae --resolution 256 --save_path banks/celeb_vae_256.pt
sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompt_file prompts/faces.txt --w 0 1 4 --sigma 0.25
sbatch runpygpu_chip.sh clip_tokens.py --bank banks/celeb_vae_256.pt
python -m pytest test_cfdm.py test_pyramid.py test_heatmap.py
```

`--sigma` is a fraction of the median pairwise distance between bank points. At about 0.1 or below, samples are copies of training images. From about 0.25 up, samples are barycenters of M-tuples of training points.

Averaged samples can look blurry. `--unsharp_amount 1.0` (with `--unsharp_radius`, a Gaussian sigma in pixels, and `--unsharp_threshold`) adds each row again after an unsharp mask, directly under the original, in both the saved grids and wandb. Metrics are still computed on the original samples.

## Coarse-to-fine pyramid

`--method pyramid` combines the CFDM with the single-image closed-form patch sampler of Qiu et al. ("Efficient and training-free single-image diffusion models", the `efficient-SID-code` submodule), and uses CLIP text heatmaps (Gandelsman et al., "Interpreting CLIP's image representation via text-based decomposition", the `clip_text_span` submodule) to choose patches. Averaging a few training images is plausible at low resolution but ghosts the detail; snapping to real patches gives sharp detail but no global layout. So each part handles the frequencies it is good at:

1. **Layout.** The bank is downsampled once to the coarsest level, which has short side `--coarse_size` (16; levels halve from full resolution down to it). The smoothed, text-conditioned CFDM runs there over the prompt's top-k matches, with `--sigma` measured at that resolution.
2. **Detail.** Each finer level runs `--pyramid_steps` (10) DDIM steps from pure noise. The denoiser is the closed-form one over every stride-1 `--patch_size` (7) patch of the prompt's best `--patch_images` (50) CLIP matches: softmax_j(−‖q − t·y_j‖²/2(1−t)² + bias_j) averaged over the clean patches y_j. Overlapping estimates are folded back with a Gaussian window (`--fold_rho` 0.2).
3. **Blend.** After every denoiser call except the last, the estimate keeps only its own detail band, x̂ − up(down(x̂)), on top of the upsampled finished coarser level. The patches can therefore sharpen the CFDM's layout but never move it. The noise estimate is taken before the blend, and `--eta` > 0 adds fresh noise, both as in their sampler.

Each patch's bias is w · log p(parent image | prompt), plus `--heatmap_lambda` times its heatmap score. The heatmaps come from the bank's CLIP token embeddings, which `clip_tokens.py --bank ...` stores in `<bank>_tokens.pt` (float16, about 0.4MB per image: 11GB for 28k images; `sample.py` memory-maps it, or point `--tokens` at it). Token (i,j) is the direct contribution of the (i,j) cell of the image (16×16 cells for ViT-L/14) to its CLIP embedding, through the last `--n_layers` (4) attention layers, with the final LayerNorm linearized, in the joint space where text lives (`heatmap.py`, a reimplementation of their `prs_hook` on the bank's HF CLIP). Its heat for a prompt is ⟨token, text⟩, minus the mean over all prompts and `--heatmap_baseline` texts. CLIP sees the same center crop that `embed.py` resizes into X, so token cells line up with X at any resolution and in any latent space. At each level, every CLIP token maps to the denoiser patches that overlap it (`token_overlap`, `token_to_patches` in `pyramid.py`). A patch's score is the overlap-weighted mean heat of the tokens it covers, standardized over the patch bank. `--heatmap_keep` < 1 keeps that fraction of the tokens, those with the highest heat over the patch images, and then every patch sharing at least one pixel with a kept token, plus `--heatmap_random` of the other patches. Because a patch usually overlaps several tokens, more patches survive than tokens, especially at the coarse levels where tokens are only a pixel or two across. Keep some of the rest: every output region snaps to some patch, so a bank of only text-relevant patches tiles them everywhere. Patch banks are built once per (prompt, w), and `patches_kept` in `metrics.json` reports the fraction kept.

At w=0 the CFDM runs over the whole downsampled bank, and the patches come from `--patch_images` uniformly random images with no text bias. Cost grows with the number of patches times the number of query patches: 50 images at 64×64 give 168k patches. At 256px, use fewer `--patch_images`.

```bash
sbatch runpygpu_chip.sh sample.py --bank banks/celebahq_pix64.pt --method pyramid --prompt_file prompts/celeb.txt --w 0 1 4
sbatch runpygpu_chip.sh sample.py --bank banks/celebahq_pix64.pt --method pyramid --heatmap_keep 0.25 --heatmap_random 0.25 --heatmap_lambda 1 --prompt_file prompts/celeb.txt --w 0 1 4
```

## Evaluation

`sample.py --fid_samples N` generates N extra samples per w, spread across the prompts, and reports:

- `fid_test` / `kid_test`: FID and KID against held-out images (`embed.py --holdout K`), which the samplers never see. KID is unbiased at a few hundred samples; FID is biased upward at small N, so only compare runs with the same N.
- `fid_train` / `kid_train`: the same against training images. A sampler that copies training images scores very well here. Read it next to `fid_test` and `fid_copy_rate`: if it's much lower than `fid_test`, the sampler is memorizing.
- `fid_copy_rate` / `fid_nn_dist`: the fraction of samples within `--copy_threshold` of a training image, and the mean distance to the nearest one, over all N samples.

The same pass also reports:

- `precision_*` / `recall_*`: k-NN precision (are samples realistic?) and recall (do they cover the data?) in the same Inception space as FID.
- `dino_nn_sim` / `dino_copy_rate`: cosine similarity to the nearest training image in DINOv2 space, and the fraction above `--dino_copy_threshold`. This catches near-copies that the pixel metric misses. Training features are cached next to the bank.
- `fid_prompt_accuracy`: the fraction of samples whose best-matching prompt, scored by the independent `--eval_clip_model`, is their own.
- `fid_lpips_diversity`: mean pairwise LPIPS between samples of the same prompt.

Per-prompt metrics also include `diversity` (pixel distance), `lpips_diversity` and `prompt_accuracy`. FID/KID need `pip install torch-fidelity`.

**Baselines.**
- `--method cfdm`: the whole-image CFDM.
- Trained diffusion models (`pretrained.py`, `finetune.py`); see the next section.
- `agreement.py`: Table 1 of Lukoianov et al. It runs `ideal`, `cfdm` and `ls` (Kamb & Ganguli's local score machine, with their CelebA-64 patch schedule) over the whole training set from the same noise as the trained network, and reports r² and MSE against it. Other trained networks (`--other_models`) give the ceiling.

## Experiments: CelebA-HQ and trained baselines

The experiments use `korexyz/celeba-hq-256x256` (28k train / 2k validation) with BLIP captions from `caption.py`. `embed.py --test_split validation` stores the held-out split as `X_test`, the FID reference. The faces are represented in four spaces: pixels at 64×64 and at 256×256, the VQ latents of `CompVis/ldm-celebahq-256`, and the FLUX.2 autoencoder's latents. `embed.py` can also build banks from other datasets: `--label_column` makes captions from class names, and `--metadata_columns` with `caption.py --caption_template` adds dataset columns to the BLIP caption.

`--space vq` encodes images with the VQ autoencoder of `CompVis/ldm-celebahq-256`, so the closed-form samplers work in the same 3×64×64 latent space as that LDM.

`--space flux2` encodes images with the FLUX.2 autoencoder (`AutoencoderKLFlux2` in the `vae` subfolder of `black-forest-labs/FLUX.2-dev`; needs diffusers ≥ 0.36). The repo is gated: accept its license on the Hub and `huggingface-cli login` on the cluster first. Latents are normalized exactly as `Flux2Pipeline` does: 2×2-patchified to 128 channels, then shifted and scaled by the VAE's BatchNorm running statistics. They are stored un-patchified, as 32×(H/8)×(W/8) (32×32×32 at 256px), which holds the same values on the finer grid and leaves the pyramid sampler more levels. Image sides must be divisible by 16. `celebahq_flux2` is embedded in bf16 and has one trained net: the `ldm-celebahq-256` UNet with 32 input/output channels at 32×32, trained from scratch (`finetune.py --channels 32 --sample_size 32`).

`pretrained.py` wraps `google/ddpm-cifar10-32`, `google/ddpm-celebahq-256` and `CompVis/ldm-celebahq-256` (noise-predicting `UNet2DModel`s) in this repo's convention z = t·x + (1−t)·ε. The noisy state x_k = z/√(t²+(1−t)²) has √ᾱ = t/√(t²+(1−t)²), the timestep is interpolated from the checkpoint's schedule, and x̂₀ = (x_k − √(1−ᾱ)·ε̂)/√ᾱ. That way the networks and the closed-form samplers share the time grid and the starting noise. Sampling starts at the checkpoint's noisiest training point (`t_min`: 0.006 for the DDPMs, 0.068 for the LDM).

`--cross_attention` attaches a cross-attention adapter (to CLIP text tokens) after every self-attention layer. Its output projection starts at zero, so the model initially matches the checkpoint. `finetune.py` trains with the checkpoint's own objective (noise prediction), dropping captions for classifier-free guidance:
- `--init pretrained`: fine-tunes the checkpoint.
- `--init scratch`: trains the same architecture from random weights on exactly our bank.

`sample.py --method net --model ...` takes `pretrained:<repo>` (the checkpoint as-is, unconditional) or a `finetune.py` checkpoint, where w is the CFG scale. `agreement.py` compares every method against a target network from the same noise.

`scripts/gen_experiments.py build|train|sample_closed|sample_nets|agreement|summarize [setting]` prints the cluster jobs for the four CelebA-HQ settings:

- `celebahq_pix64`: 64×64 pixels, the setting of Lukoianov et al. No 64px checkpoint exists, so its text-conditional nets use the `google/ddpm-celebahq-256` architecture run at 64×64 (`finetune.py --sample_size 64`; the UNet is fully convolutional). One is trained from scratch (the agreement target) and one is fine-tuned from the 256px weights.
- `celebahq_pix256`: 256×256 pixels, with `google/ddpm-celebahq-256` as-is, fine-tuned and from scratch. The 11GB float16 bank stays in CPU memory.
- `celebahq_ldm`: the 3×64×64 VQ latents, with `CompVis/ldm-celebahq-256` as-is, fine-tuned and from scratch.
- `celebahq_flux2`: the 32×32×32 FLUX.2 latents, with one scratch net.

`build` embeds, captions and computes the CLIP tokens. All four banks share one token file, `banks/celebahq_tokens.pt`, since CLIP sees the original images in every space. `sample_closed` runs the whole-image `cfdm` over the top-1000 matches (except at 256px, where it would need the bank on the GPU), plus five pyramid variants (`pyramid_variants`): no heatmap, heatmap bonus only, hard filter to the top quarter of tokens, that plus a random quarter of the other patches, and that filter plus the bonus. The pyramid geometry is per setting: 16 → 64 with 7×7 patches for 64px pixels and the VQ latents, 16 → 256 with 7×7 patches from 10 images at 256px, and 8 → 32 with 5×5 for the FLUX.2 latents. `sample_nets` runs the trained nets. Runs with a `metrics.json` already on disk are skipped. `summarize.py` collects the runs into `summary.md`/`summary.csv` and side-by-side images.

## Mechanistic pilot: do closed-form machines predict a trained network?

Following Kamb & Ganguli (*An analytic theory of creativity in convolutional diffusion models*), with text conditioning added:

1. `caption.py`: captions each bank image with BLIP and stores `captions` and their CLIP text embeddings `C` in the bank.
2. `train.py`: trains a flow-matching UNet (`textunet.py`) on exactly the bank's latents, with the same z_t = t·x + (1−t)·ε convention. It's conv-only by default, with the caption embedding added to the time embedding; `--attention` switches to self- and cross-attention over CLIP token states. Captions are dropped 10% of the time for CFG, and EMA weights are saved.
3. `els.py`: the closed-form denoisers. `ideal` is the whole-image posterior mean. `ls` uses a local P×P patch at the same position for every pixel. `els` is equivariant: P×P patches from any position. Each takes a text prior over training images (`--prior caption` for similarity to their captions, or `image` for similarity to their CLIP image embeddings).
4. `compare.py`: runs the network, feeds each z_t on its trajectory to every machine and patch size, and records r² against the network's x̂₀. It picks the best P per step on the even seeds and reports r² on the odd seeds, then compares final samples from the same noise. Outputs are `rows.csv`, `summary.json`, `r2_vs_t.png` and `finals.png`.

`python scripts/gen_mech.py caption|train|compare` prints the cluster commands for each stage.

ELS cost grows with P² × `--els_images` × H × W, so it gets its own, smaller `--els_patch_sizes`.

## Caveats

- Samples are still combinations of training points, now chosen by the text. Prompts that nothing in the dataset matches can't be produced.
- The image-distance term sharpens like $1/(1-t)^2$, so the text mostly acts early in the trajectory. That makes `start_t` and `w` the main knobs to tune.
