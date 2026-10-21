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
- `sample.py`: samples for a list of prompts and a sweep over `w`. It saves image grids, where each row is the top CLIP-retrieved training image followed by the samples, and writes `metrics.json` with CLIP score, copy rate and nearest-neighbour distance.
- `test_cfdm.py`: checks the score against autograd of the conditional mixture, plus the limits in `w`, top-k and sampling.

## Usage

```bash
git submodule update --init
sbatch runpygpu_chip.sh embed.py --dataset jlbaker361/celeb --space vae --resolution 256 --save_path banks/celeb_vae_256.pt
sbatch runpygpu_chip.sh sample.py --bank banks/celeb_vae_256.pt --prompts "a smiling woman with blonde hair" "a man with a beard" --w 0 1 4 --sigma 0.25
python -m pytest test_cfdm.py
```

`--sigma` is a fraction of the median pairwise distance between bank points. At about 0.1 or below, samples are copies of training images. From about 0.25 up, samples are barycenters of M-tuples of training points.

## Caveats

- Samples are still combinations of training points, now chosen by the text. Prompts that nothing in the dataset matches can't be produced.
- The image-distance term sharpens like $1/(1-t)^2$, so the text mostly acts early in the trajectory. That makes `start_t` and `w` the main knobs to tune.
