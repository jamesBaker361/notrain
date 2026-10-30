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
- `bank.py`: CLIP and VAE encode/decode helpers.
- `embed.py`: builds a bank: `X` (flattened pixels or SD-VAE latents) and `T` (normalized CLIP image embeddings).
- `sample.py`: samples for each prompt in `--prompt_file` (one per line; `--prompts` overrides it) and a sweep over `w`. It saves image grids, where each row is the top CLIP-retrieved training image followed by the samples, and writes `metrics.json` with CLIP score, copy rate and nearest-neighbour distance.
- `test_cfdm.py`: checks the score against autograd of the conditional mixture, plus the limits in `w`, top-k and sampling.

## Usage

```bash
git submodule update --init
sbatch runpygpu_chip.sh embed.py --dataset korexyz/celeba-hq-256x256 --test_split validation --space vae --resolution 256 --save_path banks/celeb_vae_256.pt
sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompt_file prompts/faces.txt --w 0 1 4 --sigma 0.25
python -m pytest test_cfdm.py
```

`--sigma` is a fraction of the median pairwise distance between bank points. At about 0.1 or below, samples are copies of training images. From about 0.25 up, samples are barycenters of M-tuples of training points.

Averaged samples can look blurry. `--unsharp_amount 1.0` (with `--unsharp_radius`, a Gaussian sigma in pixels, and `--unsharp_threshold`) adds each row again after an unsharp mask, directly under the original, in both the saved grids and wandb. Metrics are still computed on the original samples.

## Multiscale patches

`--scales 1 2 4` cuts each latent into a 1×1, 2×2 and 4×4 grid. Each patch region gets its own closed-form k_t over patches from the CLIP-retrieved images, with each patch keeping its parent image's text prior. The per-scale k_t's are blended over time by `--schedule` (`coarse_to_fine` by default, with linear hat weights), so different regions can come from different images. This is a local-score heuristic: only scale 1 is the exact score of a distribution.

- `--patch_positions any|same`: whether a region can take patches from every grid position or only its own. On CelebA, `any` scrambles faces (about half of the patches come from a different position), while `same` gives aligned collages.
- `--sigma` smooths only the whole-image scale. `--patch_sigma` (default 0) smooths the patch scales. Smoothing at the patch scales averages several patches together and turns the image to mush, so leaving it at 0 makes each region snap to one real, sharp patch.
- The start at `start_t` comes from whichever scale has the most weight at that moment. If that's scale 1, one whole image is noised, and by about t=0.2 the sample has locked onto it, so later patch scales just reassemble that image. If it's a patch scale (for example `--scales 2 4`), each region starts from its own patch, so samples mix images at any `start_t`.

## CLIP tree

`tree.py` builds a hierarchical index over a bank. It runs PCA on the CLIP image embeddings (`--pca_dim`), then clusters in that space with one of three `--cluster_method`s. `--cluster_on caption` builds it from the CLIP text embeddings of the bank's captions instead, and `--cluster_on mix` from normalize((1−a)·image + a·caption) with a = `--caption_weight`. Node means, and so the descent's prompt similarity, use the same embeddings (text-to-text for captions). Template captions (CIFAR-10's "a photo of a {class}") give only one distinct embedding per class, so use `mix` there. In `scripts/gen_experiments.py`, add `<method>_caption` or `<method>_mix` to `cluster_methods`.

- `kmeans` (default): recursive k-means, with k fixed (`--branching 10`) or picked per node by silhouette score (`--branching silhouette`).
- `dbscan`: recursive DBSCAN. eps is set per node as `--dbscan_quantile` of the distances to each point's `--dbscan_min_samples`-th neighbour, because a fixed eps stops working as nodes get smaller and denser. Noise points join the cluster of their nearest clustered point (`--dbscan_noise nearest`); `group` makes them one extra child instead. Nodes where DBSCAN finds fewer than two groups are split with 2-means.
- `agglomerative`: one dendrogram over all images (scipy linkage, `--linkage ward|average|complete|single`), used as the tree directly. Internal nodes are therefore binary, so depth cutoffs mean much bigger nodes than with k-means at the same depth. It needs O(N²) memory, about 3GB for 28k images and 10GB for CIFAR-10's 50k. Nodes with `--c_build` or fewer images get one leaf per image. Each node stores its mean CLIP embedding, and each internal node also stores the per-dimension mean and variance of its members' VAE latents.

`sample.py --method tree --tree <path>` walks down the tree, picking a child with probability softmax(w · logit_scale · mean CLIP similarity to the prompt). So w=0 is a uniform random walk, w=1 uses CLIP's own temperature, and large w is greedy. The walk stops at depth `--tree_depth` or at the first node with `--c_search` or fewer images, then samples from that node using `--node_sampler`:

- `diagonal`: N(mean, diag var). Dimensions are sampled independently, which decodes to speckle noise.
- `lowrank` (default): mean + Σ_i a_i (x_i − mean)/√(n−1) with a ~ N(0, I). It has the same per-dimension variance as `diagonal`, but keeps the correlations between dimensions, which gives coherent new faces.
- `cfdm`: smoothed CFDM over the node's members (uses `--sigma`, `--M`, `--steps`, `--start_t`).

- `locality`: the Wiener-mask analytical model of Lukoianov et al. (*Locality in Image Diffusion Models Emerges from Data Statistics*, Algorithm 1), with the node's images as the dataset and the node's covariance as the statistics (`locality.py`). Each output pixel q is a softmax over the node's images, using a masked distance that only counts the pixels in row q of the binarized Wiener matrix W_t = U·diag(λ/(λ+σ_t²))·Uᵀ (threshold `--tau`, relative to each row's max). Sampling starts from pure noise, with `--steps` Euler steps (the paper uses 10).
- `wiener`: their linear baseline, mean + W_t (z/t − mean), using the node's statistics.
- `curve` (`curve.py`): a 1-D curve through the node's images, sampled along. The images are put in order by an open traveling-salesman path. `--tsp lkh` uses LKH (`pip install elkai`); otherwise chained 2-opt is used, which is 2-opt with double-bridge kicks, the kick-and-reoptimize loop of Chained Lin-Kernighan. Each image's parameter t is its cumulative path length, scaled to [0,1]. A B-spline x(t) is fitted with knots at the images' t values. `--curve_mode random` (default) draws t, weighted by the CLIP text prior fitted along the curve with weight w, and returns x(t). `flow` instead runs the ideal-denoiser flow toward points on the curve, like CFDM but over the curve rather than the images. By default the curve is the polyline through the images (`--curve_degree 1`, `--curve_knots 0` = one per image), so samples are images and crossfades between neighbours on the path. Cubic curves (`--curve_degree 3`) overshoot between dissimilar images and decode to noise in latent space, and fewer knots average many images into blur. Nodes with more than `--curve_max_points` (2000) images use a random subset of that size. Each node's curve is built once and cached; a 2000-image path takes about 5 s.

`sample.py --method topk --top_k K --node_sampler <sampler>` skips the tree. The node is simply the prompt's K best CLIP matches (prompt text embedding vs image embeddings), and any of the samplers above runs on it. Each image is weighted by softmax(w · log p(image | prompt)):
- `cfdm` and `curve` add w · log p to their own image or curve weights.
- `wiener` and `locality` use the weighted mean and covariance of the K images. The covariance is still shrunk toward the global one by `--wiener_shrinkage`, because K images give rank K−1.

At w=0 the K images are used unweighted, so the prompt only acts through which images are chosen. By contrast, `--method cfdm` with `--top_k` uses the whole bank at w=0. K plays the role of the tree's depth cutoff: small K is local and copy-prone, large K is closer to the whole bank. Wiener and locality statistics depend on the prompt and w, so each (prompt, w) pair computes its own d×d eigendecomposition, once. The `sample_topk` stage of `scripts/gen_experiments.py` runs every sampler at K = 50, 200 and 1000 (`topk_values`). The runs are named `<setting>_<sampler>_K<K>` and labelled "`<sampler>` K`<K>` (no tree)" by `summarize.py`.

The masks only become local at low noise if the covariance is full rank. A node with n images has rank n−1 (in 4096 latent dimensions), so on its own the masks stay at 65–90% of the image and you get exact copies. `--wiener_shrinkage` blends the node covariance toward the full-bank one (full rank for the 28k CelebA bank, as in the paper), and `--wiener_ridge` (default 0.01) adds a fraction of the mean variance to the diagonal. On a 400-image test, ridge 0.01 gave near-copies with local edits, while 0.1 gave novel but patchy mosaics. Each node's eigendecomposition is computed once when first visited and then cached.

**Pixel space.** Everything above (tree, all node samplers, `sample.py`) also works on a pixel bank: `embed.py --space pixel --resolution 64` stores flattened RGB in [-1, 1], with d = 3·64·64 = 12,288, and needs no VAE. Keep the resolution at 64 or below (the paper uses 64×64 CelebA-HQ): the locality sampler works with d×d matrices, which is 600MB at 64px and 155GB at 256px. When a node has fewer images than dimensions and there's no shrinkage, its statistics come from an SVD of the node's images rather than a d×d eigendecomposition. In a 32px test this ran in about 0.1–0.2 s per sample on CPU. With `--wiener_shrinkage`, the global covariance is eigendecomposed once per run. Each node is then a low-rank Woodbury update of it, and all nodes step together so they share the global part of W_t. This is exact (tested against the per-node eigendecomposition, which is still available as `--wiener_exact_eig`). On CPU with 6 nodes of 20–200 images and 10 steps, `locality` went from 16.8 s to 6.1 s at d=3072, and from 177 s to 35 s at d=6144, with the one-time global eigendecomposition included. Peak GPU memory at 64px is about 4GB.

```bash
sbatch runpygpu_chip.sh embed.py --dataset korexyz/celeba-hq-256x256 --test_split validation --space pixel --resolution 64 --save_path banks/celeb_pix64.pt
sbatch runpygpu_chip.sh tree.py --bank banks/celeb_pix64.pt --save_path banks/celeb_pix64_tree.pt --branching silhouette
sbatch runpygpu_chip.sh sample.py --bank banks/celeb_pix64.pt --method tree --tree banks/celeb_pix64_tree.pt --node_sampler locality --steps 10 --wiener_shrinkage 0.5 --wiener_ridge 0 --c_search 200 --prompt_file prompts/celeb.txt --w 0 1 4
```

A walk that ends on a single-image leaf just returns that image.

## Evaluation

`sample.py --fid_samples N` generates N extra samples per w, spread across the prompts, and reports:

- `fid_test` / `kid_test`: FID and KID against held-out images (`embed.py --holdout K`), which are never in the tree or the samplers. KID is unbiased at a few hundred samples; FID is biased upward at small N, so only compare runs with the same N.
- `fid_train` / `kid_train`: the same against training images. A sampler that copies training images scores very well here. Read it next to `fid_test` and `fid_copy_rate`: if it's much lower than `fid_test`, the sampler is memorizing.
- `fid_copy_rate` / `fid_nn_dist`: the fraction of samples within `--copy_threshold` of a training image, and the mean distance to the nearest one, over all N samples.

The same pass also reports:

- `precision_*` / `recall_*`: k-NN precision (are samples realistic?) and recall (do they cover the data?) in the same Inception space as FID.
- `dino_nn_sim` / `dino_copy_rate`: cosine similarity to the nearest training image in DINOv2 space, and the fraction above `--dino_copy_threshold`. This catches near-copies that the pixel metric misses. Training features are cached next to the bank.
- `fid_prompt_accuracy`: the fraction of samples whose best-matching prompt, scored by the independent `--eval_clip_model`, is their own.
- `fid_lpips_diversity`: mean pairwise LPIPS between samples of the same prompt.

Per-prompt metrics also include `diversity` (pixel distance), `lpips_diversity` and `prompt_accuracy`. FID/KID need `pip install torch-fidelity`.

**Baselines.**
- `--node_sampler ls`: Kamb & Ganguli's local score machine, with the CelebA-64 patch schedule from Lukoianov et al.
- Trained diffusion models (`pretrained.py`, `finetune.py`); see the next section.
- `agreement.py`: the paper's Table 1. It runs `ideal`, `cfdm`, `wiener`, `locality` and `ls` over the whole training set from the same noise as the trained network, and reports r² and MSE against it. A second network trained with another seed (`--model2`) gives the ceiling.

## Standard datasets and trained baselines

The experiments use standard Hugging Face datasets with their official held-out splits: `uoft-cs/cifar10` (50k train / 10k test) and `korexyz/celeba-hq-256x256` (28k train / 2k validation). `embed.py --test_split` stores the held-out split as `X_test` (the FID reference). `--label_column label --caption_template "a photo of a {}"` makes captions from class names; CelebA-HQ gets BLIP captions from `caption.py`.

Three more 64×64 pixel settings add more varied captions:

- `imagenet_pix64`: 30k shuffled images of `benjamin-paine/imagenet-1k-64x64`, with 2k from its validation split held out. `--shuffle` is needed because the dataset is sorted by class.
- `wikiart_pix64`: the first 20 of `huggan/wikiart`'s 72 shards (already shuffled), about 22k paintings and 9GB, with 1k held out. `--data_files` loads only those shards.
- `league_pix64`: the 2,000 splash arts of `jlbaker361/league-splash-tagged`, with 200 held out.

`embed.py --metadata_columns` keeps dataset columns (class labels converted to readable names) in `bank["metadata"]`. `caption.py --caption_template` combines them with the BLIP caption, e.g. `"{blip}, {style} {genre} by {artist}"` for WikiArt and `"{blip}, {tag} {champion} splash art"` for League. A template without `{blip}` skips BLIP. These settings have no pretrained checkpoint, so their only net is the `google/ddpm-celebahq-256` architecture at 64×64 with cross-attention, trained from scratch. League's 200 held-out images make its FID noisy.

The same three datasets also run at 256×256 pixels (`imagenet_pix256`, `wikiart_pix256`, `league_pix256`) and in the 4×32×32 latents of the Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`, `*_ldm`). The CelebA-HQ LDM's VQ autoencoder was trained only on faces, so these settings don't use it. They are set up like `celebahq_pix256` and `celebahq_ldm`: same samplers and depths, a float16 CPU bank and low-rank Wiener at 256px, and LDM training on `runpygpu_chip_heavy.sh`. Each has one scratch net. For `*_ldm` that net is the `ldm-celebahq-256` UNet with 4 input/output channels at 32×32 (`finetune.py --channels 4 --sample_size 32`). ImageNet at 256px comes from `benjamin-paine/imagenet-1k-256x256`: a 30k bank plus 2k held out, both from its validation split, which is 0.7GB and not class-sorted, instead of the 20GB train split. 

`--space vq` encodes images with the VQ autoencoder of `CompVis/ldm-celebahq-256`, so the closed-form samplers work in the same 3×64×64 latent space as that LDM.

`pretrained.py` wraps `google/ddpm-cifar10-32`, `google/ddpm-celebahq-256` and `CompVis/ldm-celebahq-256` (noise-predicting `UNet2DModel`s) in this repo's convention z = t·x + (1−t)·ε. The noisy state x_k = z/√(t²+(1−t)²) has √ᾱ = t/√(t²+(1−t)²), the timestep is interpolated from the checkpoint's schedule, and x̂₀ = (x_k − √(1−ᾱ)·ε̂)/√ᾱ. That way the networks and the closed-form samplers share the time grid and the starting noise. Sampling starts at the checkpoint's noisiest training point (`t_min`: 0.006 for the DDPMs, 0.068 for the LDM).

`--cross_attention` attaches a cross-attention adapter (to CLIP text tokens) after every self-attention layer. Its output projection starts at zero, so the model initially matches the checkpoint. `finetune.py` trains with the checkpoint's own objective (noise prediction), dropping captions for classifier-free guidance:
- `--init pretrained`: fine-tunes the checkpoint.
- `--init scratch`: trains the same architecture from random weights on exactly our bank.

`sample.py --method net --model ...` takes `pretrained:<repo>` (the checkpoint as-is, unconditional) or a `finetune.py` checkpoint, where w is the CFG scale. `agreement.py` compares every method against a target network from the same noise.

`scripts/gen_experiments.py build|train|sample_tree|sample_nets|agreement|summarize [setting]` runs four settings: `cifar10_pix32`, `celebahq_pix256`, `celebahq_ldm` and `celebahq_pix64`. The last is CelebA-HQ faces at 64×64 pixels, the locality paper's setting. No 64px checkpoint exists, so its text-conditional nets use the `google/ddpm-celebahq-256` architecture run at 64×64 (`finetune.py --sample_size 64`; the UNet is fully convolutional). One is trained from scratch (the agreement target) and one is fine-tuned from the 256px weights. Each has `cfdm`/`locality`/`wiener`/`ls` at depths 0, 2, 4 and 8, a retrieval baseline (depth 99), and three trained nets (as-is, fine-tuned, scratch). `celebahq_pix256` keeps its 11GB float16 bank in CPU memory and skips `locality` (d×d masks at d = 196,608) and depth 0. Its `wiener` runs with `--wiener_rank 512`: node and global covariances are rank-k (randomized SVD, streamed from the bank) plus an isotropic term holding the left-out variance. The filter then comes from Woodbury with a small solve, so no d×d matrix is ever built. With enough components it equals the exact filter (tested), and smaller ranks smooth the samples, so check sensitivity to the rank (e.g. 512 vs 2048). `cluster_methods` at the top of the script picks which trees to build and sample. k-means keeps the original names; other methods get `_<method>` suffixes. The `tree` stage builds only the trees, for banks that already exist. `summarize.py` collects the runs into `summary.md`/`summary.csv` and side-by-side images. On macOS, run `tree.py` with `OMP_NUM_THREADS=1`; otherwise sklearn and torch can load clashing OpenMP runtimes and segfault.

```bash
sbatch runpygpu_chip.sh tree.py --bank banks/celeb.pt --save_path banks/celeb_tree.pt --branching silhouette
sbatch runpygpu_chip.sh sample.py --bank banks/celeb.pt --method tree --tree banks/celeb_tree.pt --node_sampler lowrank --prompt_file prompts/celeb.txt --w 0 1 4
```

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
