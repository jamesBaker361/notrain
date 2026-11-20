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
- `heatmap.py`: CLIP text heatmaps over image tokens, used to weight and filter the pyramid's patches.
- `bank.py`: CLIP and VAE encode/decode helpers.
- `embed.py`: builds a bank: `X` (flattened pixels or SD-VAE latents) and `T` (normalized CLIP image embeddings).
- `sample.py`: samples for each prompt in `--prompt_file` (one per line; `--prompts` overrides it) and a sweep over `w`, with `--method cfdm`, `pyramid` or `net`. It saves image grids, where each row is the top CLIP-retrieved training image followed by the samples, and writes `metrics.json` with CLIP score, copy rate and nearest-neighbour distance.
- `test_cfdm.py`: checks the score against autograd of the conditional mixture, plus the limits in `w`, top-k and sampling. `test_pyramid.py` and `test_heatmap.py` cover the other two samplers' pieces.

## Usage

```bash
git submodule update --init
sbatch runpygpu_chip.sh embed.py --dataset korexyz/celeba-hq-256x256 --test_split validation --space vae --resolution 256 --save_path banks/celeb_vae_256.pt
sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompt_file prompts/faces.txt --w 0 1 4 --sigma 0.25
python -m pytest test_cfdm.py test_pyramid.py test_heatmap.py
```

`--sigma` is a fraction of the median pairwise distance between bank points. At about 0.1 or below, samples are copies of training images. From about 0.25 up, samples are barycenters of M-tuples of training points.

Averaged samples can look blurry. `--unsharp_amount 1.0` (with `--unsharp_radius`, a Gaussian sigma in pixels, and `--unsharp_threshold`) adds each row again after an unsharp mask, directly under the original, in both the saved grids and wandb. Metrics are still computed on the original samples.

## Coarse-to-fine pyramid

`--method pyramid` combines the CFDM with the single-image closed-form patch sampler of Qiu et al. ("Efficient and training-free single-image diffusion models", the `efficient-SID-code` submodule), and uses CLIP text heatmaps (Gandelsman et al., "Interpreting CLIP's image representation via text-based decomposition", the `clip_text_span` submodule) to choose patches. Averaging a few training images is plausible at low resolution but ghosts the detail; snapping to real patches gives sharp detail but no global layout. So each part handles the frequencies it is good at:

1. **Layout.** The bank is downsampled once to the coarsest level, which has short side `--coarse_size` (16; levels halve from full resolution down to it). The smoothed, text-conditioned CFDM runs there over the prompt's top-k matches, with `--sigma` measured at that resolution.
2. **Detail.** Each finer level runs `--pyramid_steps` (10) DDIM steps from pure noise. The denoiser is the closed-form one over every stride-1 `--patch_size` (7) patch of the prompt's best `--patch_images` (50) CLIP matches: softmax_j(−‖q − t·y_j‖²/2(1−t)² + bias_j) averaged over the clean patches y_j. Overlapping estimates are folded back with a Gaussian window (`--fold_rho` 0.2).
3. **Blend.** After every denoiser call except the last, the estimate keeps only its own detail band, x̂ − up(down(x̂)), on top of the upsampled finished coarser level. The patches can therefore sharpen the CFDM's layout but never move it. The noise estimate is taken before the blend, and `--eta` > 0 adds fresh noise, both as in their sampler.

Each patch's bias is w · log p(parent image | prompt), plus `--heatmap_lambda` times its heatmap score. The score is the patch footprint's mean over the parent image's text heatmap ⟨c_token_i, text⟩: the direct contribution of each CLIP token through the last `--heatmap_layers` (4) attention layers, with the final LayerNorm linearized (`heatmap.py`, a reimplementation of their `prs_hook` on the bank's HF CLIP). It is computed minus the mean heatmap over all prompts and `--heatmap_baseline` texts, then standardized over the bank. `--heatmap_keep` < 1 keeps only that fraction of patches, those with the highest scores, plus `--heatmap_random` of the rest. Keep some of the rest: every output region snaps to some patch, so a bank of only text-relevant patches tiles them everywhere. Patch banks are built once per (prompt, w), and `patches_kept` in `metrics.json` reports the fraction kept.

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

## Standard datasets and trained baselines

The experiments use standard Hugging Face datasets with their official held-out splits: `uoft-cs/cifar10` (50k train / 10k test) and `korexyz/celeba-hq-256x256` (28k train / 2k validation). `embed.py --test_split` stores the held-out split as `X_test` (the FID reference). `--label_column label --caption_template "a photo of a {}"` makes captions from class names; CelebA-HQ gets BLIP captions from `caption.py`.

Three more 64×64 pixel settings add more varied captions:

- `imagenet_pix64`: 30k shuffled images of `benjamin-paine/imagenet-1k-64x64`, with 2k from its validation split held out. `--shuffle` is needed because the dataset is sorted by class.
- `wikiart_pix64`: the first 20 of `huggan/wikiart`'s 72 shards (already shuffled), about 22k paintings and 9GB, with 1k held out. `--data_files` loads only those shards.
- `league_pix64`: the 2,000 splash arts of `jlbaker361/league-splash-tagged`, with 200 held out.

`embed.py --metadata_columns` keeps dataset columns (class labels converted to readable names) in `bank["metadata"]`. `caption.py --caption_template` combines them with the BLIP caption, e.g. `"{blip}, {style} {genre} by {artist}"` for WikiArt and `"{blip}, {tag} {champion} splash art"` for League. A template without `{blip}` skips BLIP. These settings have no pretrained checkpoint, so their only net is the `google/ddpm-celebahq-256` architecture at 64×64 with cross-attention, trained from scratch. League's 200 held-out images make its FID noisy.

The same three datasets also run at 256×256 pixels (`imagenet_pix256`, `wikiart_pix256`, `league_pix256`) and in the 4×32×32 latents of the Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`, `*_ldm`). The CelebA-HQ LDM's VQ autoencoder was trained only on faces, so these settings don't use it. They are set up like `celebahq_pix256` and `celebahq_ldm`: a float16 CPU bank at 256px, and LDM training on `runpygpu_chip_heavy.sh`. Each has one scratch net. For `*_ldm` that net is the `ldm-celebahq-256` UNet with 4 input/output channels at 32×32 (`finetune.py --channels 4 --sample_size 32`). ImageNet at 256px comes from `benjamin-paine/imagenet-1k-256x256`: a 30k bank plus 2k held out, both from its validation split, which is 0.7GB and not class-sorted, instead of the 20GB train split. 

`--space vq` encodes images with the VQ autoencoder of `CompVis/ldm-celebahq-256`, so the closed-form samplers work in the same 3×64×64 latent space as that LDM.

`pretrained.py` wraps `google/ddpm-cifar10-32`, `google/ddpm-celebahq-256` and `CompVis/ldm-celebahq-256` (noise-predicting `UNet2DModel`s) in this repo's convention z = t·x + (1−t)·ε. The noisy state x_k = z/√(t²+(1−t)²) has √ᾱ = t/√(t²+(1−t)²), the timestep is interpolated from the checkpoint's schedule, and x̂₀ = (x_k − √(1−ᾱ)·ε̂)/√ᾱ. That way the networks and the closed-form samplers share the time grid and the starting noise. Sampling starts at the checkpoint's noisiest training point (`t_min`: 0.006 for the DDPMs, 0.068 for the LDM).

`--cross_attention` attaches a cross-attention adapter (to CLIP text tokens) after every self-attention layer. Its output projection starts at zero, so the model initially matches the checkpoint. `finetune.py` trains with the checkpoint's own objective (noise prediction), dropping captions for classifier-free guidance:
- `--init pretrained`: fine-tunes the checkpoint.
- `--init scratch`: trains the same architecture from random weights on exactly our bank.

`sample.py --method net --model ...` takes `pretrained:<repo>` (the checkpoint as-is, unconditional) or a `finetune.py` checkpoint, where w is the CFG scale. `agreement.py` compares every method against a target network from the same noise.

`scripts/gen_experiments.py build|train|sample_closed|sample_nets|agreement|summarize [setting]` prints the cluster jobs for every setting: `cifar10_pix32`, `celebahq_pix256`, `celebahq_ldm`, `celebahq_pix64`, and the ImageNet/WikiArt/League settings above. `celebahq_pix64` is CelebA-HQ faces at 64×64 pixels, the setting of Lukoianov et al. No 64px checkpoint exists, so its text-conditional nets use the `google/ddpm-celebahq-256` architecture run at 64×64 (`finetune.py --sample_size 64`; the UNet is fully convolutional). One is trained from scratch (the agreement target) and one is fine-tuned from the 256px weights.

`sample_closed` runs the whole-image `cfdm` over the top-1000 matches, plus five pyramid variants (`pyramid_variants`): no heatmap, heatmap bonus only, hard filter to the top quarter, top quarter plus a random quarter, and that filter plus the bonus. The pyramid geometry is per setting: 8 → 32 with 5×5 patches for CIFAR-10 and the 4×32×32 SD-VAE latents, 16 → 64 with 7×7 for 64px pixels and the VQ latents, and 16 → 256 with 10 patch images at 256px. The 256px settings keep their 11GB float16 bank in CPU memory and skip the whole-image `cfdm`. `sample_nets` runs the trained nets (as-is, fine-tuned, scratch). Runs with a `metrics.json` already on disk are skipped. `summarize.py` collects the runs into `summary.md`/`summary.csv` and side-by-side images.

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
