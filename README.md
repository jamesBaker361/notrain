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
sbatch runpygpu_chip.sh embed.py --dataset jlbaker361/celeb --space vae --resolution 256 --save_path banks/celeb_vae_256.pt
sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompt_file prompts/faces.txt --w 0 1 4 --sigma 0.25
python -m pytest test_cfdm.py
```

`--sigma` is a fraction of the median pairwise distance between bank points. At about 0.1 or below, samples are copies of training images. From about 0.25 up, samples are barycenters of M-tuples of training points.

## Multiscale patches

`--scales 1 2 4` cuts each latent into a 1×1, 2×2 and 4×4 grid. Each patch region gets its own closed-form k_t over patches from the CLIP-retrieved images, with each patch keeping its parent image's text prior. The per-scale k_t's are blended over time by `--schedule` (`coarse_to_fine` by default, with linear hat weights), so different regions can come from different images. This is a local-score heuristic: only scale 1 is the exact score of a distribution.

- `--patch_positions any|same`: whether a region can take patches from every grid position or only its own. On CelebA, `any` scrambles faces (about half of the patches come from a different position), while `same` gives aligned collages.
- `--sigma` smooths only the whole-image scale. `--patch_sigma` (default 0) smooths the patch scales. Smoothing at the patch scales averages several patches together and turns the image to mush, so leaving it at 0 makes each region snap to one real, sharp patch.
- The start at `start_t` comes from whichever scale has the most weight at that moment. If that's scale 1, one whole image is noised, and by about t=0.2 the sample has locked onto it, so later patch scales just reassemble that image. If it's a patch scale (for example `--scales 2 4`), each region starts from its own patch, so samples mix images at any `start_t`.

## CLIP tree

`tree.py` builds a hierarchical index over a bank. It runs PCA on the CLIP image embeddings (`--pca_dim`), then recursive k-means in that space, with k fixed (`--branching 10`) or picked per node by silhouette score (`--branching silhouette`). Nodes with `--c_build` or fewer images get one leaf per image. Each node stores its mean CLIP embedding, and each internal node also stores the per-dimension mean and variance of its members' VAE latents.

`sample.py --method tree --tree <path>` walks down the tree, picking a child with probability softmax(w · logit_scale · mean CLIP similarity to the prompt). So w=0 is a uniform random walk, w=1 uses CLIP's own temperature, and large w is greedy. The walk stops at depth `--tree_depth` or at the first node with `--c_search` or fewer images, then samples from that node using `--node_sampler`:

- `diagonal`: N(mean, diag var). Dimensions are sampled independently, which decodes to speckle noise.
- `lowrank` (default): mean + Σ_i a_i (x_i − mean)/√(n−1) with a ~ N(0, I). It has the same per-dimension variance as `diagonal`, but keeps the correlations between dimensions, which gives coherent new faces.
- `cfdm`: smoothed CFDM over the node's members (uses `--sigma`, `--M`, `--steps`, `--start_t`).

A walk that ends on a single-image leaf just returns that image. On macOS, run `tree.py` with `OMP_NUM_THREADS=1`; otherwise sklearn and torch can load clashing OpenMP runtimes and segfault.

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
