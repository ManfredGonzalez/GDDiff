# GDDiff — Spatially Weighted Guided Diffusion for Face De-Identification

> **Reconstruct, Don't Retrain: Spatially Weighted Guided Diffusion for Face De-Identification**
>
> Manfred Gonzalez-Hernandez¹, Danny Xie-Li², Fabian Fallas-Moya³, Darian Tomašević¹,
> Luka Šajn¹, Peter Peer¹, Vitomir Štruc⁴, Žiga Emeršič¹, Blaž Meden¹\*
>
> ¹ University of Ljubljana, Faculty of Computer and Information Science, Slovenia
> ² Costa Rica Institute of Technology, Cartago, Costa Rica
> ³ University of Costa Rica, Cartago, Costa Rica
> ⁴ University of Ljubljana, Faculty of Electrical Engineering, Slovenia
>
> \* Corresponding author: Blaž Meden (`blaz.meden@fri.uni-lj.si`)
>

---

## Overview

**GDDiff** (Guided De-identification Diffusion) is a **training-free** face de-identification
framework. It does not fine-tune, retrain, or condition a diffusion model, and it does not use an
identity encoder. Instead, it reshapes *where* the data-consistency guidance of a pretrained
diffusion sampler is allowed to act during reverse diffusion.

The core idea: **identity cues are spatially localized**. If the data-fidelity correction that pulls
the reverse trajectory back toward the observed image is *attenuated inside the facial region* and
*left intact everywhere else*, the sampler reconstructs a realistic image whose face drifts away
from the original identity while pose, expression, background, and global structure are preserved.

Attenuation is applied with two complementary per-step spatial weighting fields:

- a **Gaussian** field — a smooth low-pass regularizer that suppresses in-face updates → stronger
  identity suppression;
- a **Laplacian-of-Gaussian (Lpl)** field — an edge-sensitive field that amplifies guidance near
  facial boundaries → stronger structural fidelity.

These run as two parallel branches. At a user-chosen timestep `t_mu`, their predictions are averaged
to spawn a third **average-face** branch, whose in-face guidance is frozen with a zero mask. One
sampling pass therefore returns **three de-identified outputs of increasing recognizability**, and
the position of `t_mu` (exposed as `--per`, the paper's λ) is an inference-time dial on the
privacy–utility trade-off — no retraining required.

### Contributions

- A training-free de-identification framework applying spatially weighted Gaussian / Lpl guidance to
  the reverse diffusion process, with explicit control over the privacy–fidelity trade-off.
- A mathematical interpretation of the method as a **spatially preconditioned** optimization process
  (gradient-descent dynamics, variance reduction via averaged residuals, stability under a
  small-step condition).
- A **multi-branch sampler** producing three de-identified outputs per input, letting the user pick
  their preferred privacy–utility balance at inference time.
- Evaluation across four datasets (RaFD, XM2VTS, LFW, CALFW) on identity suppression, image quality,
  and data utility, with the Mean-GDDiff variants taking the **top three Friedman mean ranks** among
  13 methods.

---

## Method

### 1. Preconditioned guidance vectors

GDDiff builds on the preconditioned-guidance formulation of DDPG (Garber & Tirer, CVPR 2024), which
interpolates between back-projection (BP) and least-squares (LS) data-fidelity guidance. In the
general case:

```
g_BP = Aᵀ (A Aᵀ + η I)⁻¹ (A x_{0|t} − y)
g_LS = c Aᵀ (A x_{0|t} − y)
```

For face de-identification the relevant observation model is pure denoising (`A = I`,
`y = x* + ε`), and both terms collapse to scaled pixel-wise residuals:

```
g_BP = (1 / (1 + η)) (x_{0|t} − y)      g_LS = c (x_{0|t} − y)
```

No matrix operations remain — only pointwise differences between the model's clean-image estimate
and the observation.

### 2. Spatial weighting functions (not filters)

Given the face bounding-box center `(u_c, v_c)`, with `r² = (u − u_c)² + (v − v_c)²` and kernel
variance `σ_t²`:

```
G_t = exp(−r² / 2σ_t²)
L_t = (1 − r² / 2σ_t²) · exp(−r² / 2σ_t²)
```

`σ_t²` **grows with `t`**, progressively broadening the modulated region as reverse diffusion
unfolds. Importantly, `W_t` is **not convolved** with anything: it is a spatially varying scalar
field applied *pointwise* to the guidance vectors. The low-pass / high-pass interpretation comes
from the spatial structure of `W_t` itself (its smoothness vs. its second-derivative sign changes),
not from a filtering operation.

The spatially modulated guidance is:

```
g_t^(W) = W_t^B ⊙ [ (1 − δ_t) g_BP + δ_t g_LS ]
```

Low weights inside the face attenuate the update (promoting anonymization); high weights outside
enforce fidelity to the observation.

### 3. Multi-branch sampling and the average-face branch

Branches `B ∈ {G, L}` evolve in parallel (vectorized over the batch dimension). At `t = t_mu`, a
third branch `A` is spawned:

```
x_{0|t}^A = ½ (x_{0|t}^G + x_{0|t}^L)
ε̂_t^A    = ½ (ε̂_t^G + ε̂_t^L)
W_t^A     = Z(u,v)          # 0 inside the face bbox, 1 elsewhere
```

The zero mask `Z` cancels guidance updates *inside the face only*, keeping the facial region in a
guided transition, while averaging the **noise residuals** lets the background and global structure
of branch `A` keep evolving coherently with the other two. Without the residual averaging the branch
would structurally "stall" and produce artifacts.

The per-step update, for each branch `B ∈ {G, L, A}`:

```
x̃_{t−1}^B = x_{0|t}^B − μ_t · W_t^B ⊙ [ (1 − δ_t) g_BP + δ_t g_LS ]
ε̂_t^B     = ( x_t^B − √ᾱ_t · x̃_{t−1}^B ) / √(1 − ᾱ_t)
x_{t−1}^B  = √ᾱ_{t−1} · x̃_{t−1}^B + √(1 − ᾱ_{t−1}) · ( w_t √(1 − ζ) ε̂_t^B + √ζ ε^B )
```

### Theoretical insights

- **Gradient of a spatially weighted energy.** The update is one gradient-descent step on
  `E_t(x) = ½ κ_t ‖W_t^{1/2} (x − y)‖²` with `κ_t = (1 − δ_t)/(1 + η) + δ_t c`. Since `W_t` is PSD,
  each iteration decreases `E_t` for `μ_t ≤ 2 / (κ_t λ_max(W_t))`.
- **Spatial preconditioning.** Element-wise weighting is a diagonal metric in gradient space, i.e. a
  spatially varying step-size field: a smaller effective learning rate inside the face, a larger one
  outside. Formally, gradient descent under a locally adaptive (Riemannian) metric `P_t⁻¹`.
- **Mean-energy flow.** The averaged residual at `t_mu` follows the gradient of
  `E_t^(A) = ½ (E_t^(G) + E_t^(L))` — a midpoint trajectory in latent space.
- **Variance reduction.** Under mild independence assumptions,
  `Var[ε̂_t^A] ≈ ½ Var[ε̂_t^G]`, yielding more stable trajectories and smoother convergence.

### Spectral evidence

The Laplacian branch really does prioritize edge energy. Mean power (dB) of the masked guidance
vectors (λ = 0.5); `L*` is the Lpl "edge boost" over the Gaussian branch in the high-frequency band:

| Step `t` | Branch | LF (dB) | HF (dB) | `L*` (dB) |
|---:|:---|---:|---:|---:|
| 900 | G ⊙ E | 107.47 | 77.42 | — |
| 900 | L ⊙ E | 107.88 | 78.63 | **+1.21** |
| 800 | G ⊙ E | 104.76 | 70.07 | — |
| 800 | L ⊙ E | 106.22 | 71.91 | **+1.84** |
| 500 | G ⊙ E | 85.52 | 59.96 | — |
| 500 | L ⊙ E | 86.22 | 60.19 | **+0.23** |
| 500 | A | 85.44 | 60.20 | — |
| 100 | G ⊙ E | 60.91 | 59.11 | — |
| 100 | L ⊙ E | 61.00 | 59.03 | −0.08 |
| 100 | A | 60.84 | 59.37 | — |

G = Gaussian (low-pass regularizer), L = Laplacian-of-Gaussian (edge-sensitive), A = average branch,
E = interpolated guidance direction `[(1 − δ_t) g_BP + δ_t g_LS]`. The boost is largest early in
sampling and settles to ≈0 by `t = 100`, once the image has reached full reconstruction.

---

## Where the method lives in this code

The sampler is [functions/ddpg_scheme.py](functions/ddpg_scheme.py); the driver that detects faces
and dispatches branches is [guided_diffusion/diffusion.py](guided_diffusion/diffusion.py).

| Paper | Code |
|:---|:---|
| Weighting field `G_t` (Eq. 13) | [`create_gaussian_kernel`](functions/ddpg_scheme.py#L83) |
| Weighting field `L_t` (Eq. 13) | [`create_laplacian_kernel`](functions/ddpg_scheme.py#L112) |
| Zero mask `Z(u,v)` (Alg. 1, line 10) | [`create_zero_kernel`](functions/ddpg_scheme.py#L145) |
| Variance schedule `σ_t²` growing with `t` | [ddpg_scheme.py:236](functions/ddpg_scheme.py#L236) |
| `t_mu` from percentile λ | [`get_percentile_index`](functions/ddpg_scheme.py#L198), [ddpg_scheme.py:228](functions/ddpg_scheme.py#L228) |
| Guidance vectors `g_BP`, `g_LS` (Eq. 12) | [ddpg_scheme.py:303-304](functions/ddpg_scheme.py#L303-L304) |
| Branch stacking `W_t = concat(G_t, L_t[, Z])` | [ddpg_scheme.py:363-385](functions/ddpg_scheme.py#L363-L385) |
| Modulated update `x̃_{t−1}^B` (Eq. 15) | [ddpg_scheme.py:444](functions/ddpg_scheme.py#L444) |
| Average-face branch `x_{0\|t}^A` (Eq. 16) | [ddpg_scheme.py:273-275](functions/ddpg_scheme.py#L273-L275) |
| Averaged residual `ε̂_t^A` | [ddpg_scheme.py:503-506](functions/ddpg_scheme.py#L503-L506) |
| Branch spawn trigger at `t = t_mu` | [ddpg_scheme.py:527-528](functions/ddpg_scheme.py#L527-L528) |
| Face bounding box + centered fallback | [diffusion.py:459-483](guided_diffusion/diffusion.py#L459-L483), [`get_fallback_bbox`](guided_diffusion/diffusion.py#L26) |
| Branch → output folder mapping | [diffusion.py:588-596](guided_diffusion/diffusion.py#L588-L596) |

The main sampling entry point is
[`ddpg_diffusion`](functions/ddpg_scheme.py#L203); inside it, `k_avg` is the flag that marks the
average-face branch as active.

---

## Setup

### Environment

The project runs from the bundled conda environment. Python 3.10, PyTorch 2.x + CUDA 12.6, plus
`dlib`, `opencv-python`, `lpips`, `pytorch-lightning`:

```bash
conda env create -f DDPG_env.yml
conda activate DDPG_env
```

### Pretrained diffusion model

GDDiff runs on **frozen, pretrained** diffusion weights — nothing here is trained.

Download the CelebA-HQ 256×256 checkpoint
([link](https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view?usp=share_link))
and place it at:

```
exp/logs/celeba/celeba_hq.ckpt
```

An ImageNet 256×256 unconditional checkpoint
([link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt))
is also supported at `exp/logs/imagenet/256x256_diffusion_uncond.pt`, but all reported
de-identification results use the CelebA-HQ model.

### Face detector

Face localization uses the OpenCV Haar cascade shipped in the repo root
(`haarcascade_frontalface_default.xml`). If no face is detected, the code falls back to a centered
box covering 65% of the shorter image side ([`get_fallback_bbox`](guided_diffusion/diffusion.py#L26)),
so sampling never crashes on a detector miss. Face alignment follows the CPP-DeID protocol; see
[align_dataset.py](align_dataset.py) (dlib 68-landmark predictor, expected at
`model_dlib/shape_predictor_68_face_landmarks.dat`).

### Data layout

Inputs are read with `torchvision.datasets.ImageFolder`, so images must sit in a **class
subdirectory**:

```
exp/datasets/<path_y>/<subfolder>/*.png
```

`--path_y` names the directory under `exp/datasets/`. Images are resized to 256×256 on load.

---

## Running GDDiff

### Quick start

```bash
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0.05 \
  --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7 --per 0.5
```

This is the command in [evaluation900_Manfred.sh](evaluation900_Manfred.sh). To sweep λ, see
[run_commands2.sh](run_commands2.sh).

### Outputs

One sampling pass writes **three** de-identified images per input, one per branch:

```
exp/image_samples/<subfolder>/<per>_mean/<image>.png    # Mean-GDDiff  — average-face branch + zero mask
exp/image_samples/<subfolder>/<per>_gauss/<image>.png   # Gauss-GDDiff — Gaussian low-pass weighting
exp/image_samples/<subfolder>/<per>_Lapl/<image>.png    # Lpl-GDDiff   — edge-sensitive weighting
```

`<subfolder>` is the ImageFolder class directory name, `<per>` is the `--per` value you passed.
Branch index → folder mapping is `0 → gauss`, `1 → Lapl`, `2 → mean`
([diffusion.py:588-596](guided_diffusion/diffusion.py#L588-L596)).

Listed above in **increasing recognizability**: across all four datasets the `mean` branch has the
lowest AUC (strongest identity suppression), `gauss` is intermediate, and `Lapl` is the most
recognizable but structurally most faithful. **The `_mean` branch is the one the paper's Mean-GDDiff
numbers refer to.**

When `save_imgs` is enabled (default), a per-image diagnostics folder
`exp/image_samples/<subfolder>/<image>_<per>/` is also written, containing `x0_t`, `xt_next_tilde`,
`xt_next`, `guidance_BP`/`guidance_LS` heatmaps, the masked-guidance `.npy` dumps, and the Gaussian /
Laplacian / zero kernel heatmaps at `t = 900`. Note this folder is **deleted and recreated on every
image** — set `save_imgs = False` ([diffusion.py:544](guided_diffusion/diffusion.py#L544)) for batch
runs.

### Parameters

```
python main.py --config {config}.yml --path_y {dataset_folder} --deg {deg} --sigma_y {sigma_y} \
  -i {image_folder} --inject_noise {inject_noise} --gamma {gamma} --zeta {zeta} \
  --eta_tilde {eta_tilde} --step_size_mode {step_size_mode} --operator_imp {operator_imp} \
  --scale_ls {scale_ls} --per {lambda}
```

| Flag | Paper symbol | Meaning |
|:---|:---|:---|
| `--per` | **λ** | **The privacy–utility dial.** Percentile of the trajectory at which `t_mu` fires and the average-face branch is spawned. Paper reports λ ∈ {0.1, 0.2, 0.5}; also names the output folders. Internally `per = 1 − (--per + 0.1)` ([diffusion.py:548](guided_diffusion/diffusion.py#L548)). |
| `--eta_tilde` | η | Regularizes the pseudoinverse in `g_BP`; effective `η_reg = max(1e-4, σ_y² · eta_tilde)`. |
| `--scale_ls` | c | Least-squares guidance scale. |
| `--gamma` | γ | Controls `δ_t = ᾱ_{t−1}^γ`, the BP↔LS interpolation; larger ⇒ more BP dominance. |
| `--zeta` | ζ | Noise interpolation between the estimated residual `ε̂_t` and fresh Gaussian noise. |
| `--step_size_mode` | μ_t | `0` = fixed 1, `1` = decay `(1 − ᾱ_{t−1})/(1 − ᾱ_t)` (used here), `2` = fixed for BP, decay for LS. |
| `--sigma_y` | σ_y | Observation noise level (internally doubled to account for the `[-1, 1]` rescaling). |
| `--deg` | A | Degradation / measurement operator. |
| `--inject_noise` | — | `1` = stochastic sampler (DDPG-style), `0` = deterministic (IDPG-style). |
| `--operator_imp` | — | `FFT` or `SVD` operator implementation. |
| `--config` | — | YAML under [configs/](configs/). |
| `--path_y` | — | Dataset directory under `exp/datasets/`. |
| `-i` | — | Output subfolder name under `exp/image_samples/`. |
| `--seed` | — | RNG seed (default `1234`). |
| `--subset_start` / `--subset_end` | — | Process only a slice of the dataset. |

The number of reverse diffusion steps is set in the YAML config, not on the command line:

```yaml
sampling:
  T_sampling: <desired_sampling_steps>
```

### In-code switches

A few de-identification switches are hardcoded in
[`ddpg_wrapper`](guided_diffusion/diffusion.py#L272) rather than exposed as CLI flags. Edit them
there:

| Variable | Default | Effect |
|:---|:---|:---|
| `deid` ([:439](guided_diffusion/diffusion.py#L439)) | `True` | Master switch. `False` falls back to plain DDPG restoration. |
| `k` ([:440](guided_diffusion/diffusion.py#L440)) | `2` | Number of independent noise latents ⇒ the Gaussian and Lpl branches. |
| `gaussian_kern` ([:551](guided_diffusion/diffusion.py#L551)) | `True` | Enables the spatial weighting fields. |
| `diff_priv` ([:547](guided_diffusion/diffusion.py#L547)) | `False` | Experimental Laplace-noise obfuscation of the guidance inside the bbox. **Not part of the paper** — GDDiff makes no formal differential-privacy claim. |
| `save_imgs` ([:544](guided_diffusion/diffusion.py#L544)) | `True` | Per-step diagnostics dumps (see Outputs). |
| `only_mean` ([:545](guided_diffusion/diffusion.py#L545)) | `False` | `True` saves only the average-face branch. |

---

## Results

Setup: all generated images are 256×256. Identity similarity is pairwise cosine similarity
(normalized to `[0, 1]`) between **AdaFace** embeddings, over *mated* (same identity) and *non-mated*
(different identity) pairs. Good de-identification collapses the mated distribution into the
non-mated one, i.e. **lower AUC / F1 is better**. Image quality is Fréchet Distance on **DINOv2**
embeddings (FD ↓) and pixel MSE (↓). Data utility is gender-classification accuracy (gACC ↑).

Baselines are the SOTA methods with public implementations: FAMS, FADM, LDFA, DeepPrivacy, CLEANIR,
AMT-GAN, CPP-DeID. Closed-set methods (FALCO, RIDDLE) and methods with substantially higher
inference cost (Diff-Privacy) are excluded to keep the comparison fair and reproducible.

### Datasets

| Dataset | Condition | Content | Pairs |
|:---|:---|:---|:---|
| **RaFD** | Studio, 1024×1024 | 1,607 images, 73 participants, 3 angles | 1,608 mated / 1,608 non-mated |
| **XM2VTS** | Studio, 1024×1024 | 2,359 portraits, 295 subjects, uniform lighting | 1,472 mated / 1,472 non-mated |
| **LFW** | In-the-wild, aligned 1024×1024 | 13,233 images, 5,749 identities; 1,680 subjects with ≥2 images | 3,000 mated / 3,000 non-mated |
| **CALFW** | In-the-wild, cross-age | 11,517 aligned images, 4,025 identities, 10-fold splits | standard splits |

### Studio datasets (RaFD, XM2VTS)

Identity verification (IV), image quality (IQ), data utility (DU). Ours in **bold**.

| Method | RaFD AUC ↓ | RaFD F1 ↓ | RaFD FD ↓ | RaFD MSE ↓ | RaFD gACC ↑ | XM2VTS AUC ↓ | XM2VTS F1 ↓ | XM2VTS FD ↓ | XM2VTS MSE ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mean-GDDiff (0.1)** | 0.86 | 0.78 | 670.18 | 0.01 | 0.82 | 0.83 | 0.75 | 577.48 | 0.02 |
| **Mean-GDDiff (0.2)** | 0.82 | 0.75 | 700.49 | 0.01 | 0.81 | 0.77 | 0.71 | 626.18 | 0.02 |
| **Mean-GDDiff (0.5)** | 0.77 | 0.70 | 747.91 | 0.02 | 0.80 | **0.69** | **0.62** | 729.87 | 0.04 |
| DeepPrivacy | 0.89 | 0.81 | 448.67 | 0.01 | 0.71 | 0.92 | 0.85 | **374.91** | 0.02 |
| FAMS | 0.89 | 0.81 | 526.76 | 0.01 | 0.40 | 0.88 | 0.80 | 600.45 | 0.01 |
| **Gauss-GDDiff** | 0.93 | 0.85 | 625.97 | 0.01 | 0.81 | 0.92 | 0.85 | 539.45 | 0.01 |
| CLEANIR | 0.81 | 0.73 | 1660.59 | 0.01 | 0.67 | 0.90 | 0.82 | 1226.19 | 0.01 |
| **Lpl-GDDiff** | 0.96 | 0.90 | 556.08 | 0.01 | 0.83 | 0.97 | 0.91 | 457.76 | 0.01 |
| CPP-DeID (0.9) | **0.65** | **0.61** | 1476.96 | 0.18 | 0.62 | 0.73 | 0.67 | 943.39 | 0.09 |
| FADM | 0.94 | 0.87 | 521.99 | 0.01 | **0.85** | 0.98 | 0.93 | 474.29 | 0.02 |
| LDFA | 0.85 | 0.78 | 998.28 | 0.08 | 0.74 | 0.85 | 0.77 | 892.41 | 0.07 |
| CPP-DeID (0.5) | 0.97 | 0.92 | 900.92 | 0.06 | 0.77 | 0.99 | 0.97 | 977.95 | 0.05 |
| AMT-GAN | 0.99 | 0.96 | 985.86 | 0.02 | 0.84 | 1.00 | 0.98 | 1174.73 | 0.06 |

gACC is reported for RaFD only. Mean-GDDiff (0.5) reaches the **lowest AUC on XM2VTS** among most
competing methods. CPP-DeID (0.9) suppresses identity harder on RaFD, but at a steep cost in image
quality (FD 1476.96, MSE 0.18 vs. 747.91 / 0.02).

### In-the-wild datasets (LFW, CALFW)

| Method | LFW AUC ↓ | LFW F1 ↓ | LFW FD ↓ | LFW MSE ↓ | CALFW AUC ↓ | CALFW F1 ↓ | CALFW FD ↓ | CALFW MSE ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mean-GDDiff (0.1)** | 0.77 | 0.70 | 289.30 | 0.01 | 0.67 | 0.62 | 210.54 | 0.01 |
| **Mean-GDDiff (0.2)** | 0.73 | 0.67 | 311.48 | 0.02 | 0.64 | 0.60 | 221.91 | 0.02 |
| **Mean-GDDiff (0.5)** | 0.70 | 0.65 | 362.03 | 0.03 | 0.62 | 0.58 | 250.10 | 0.03 |
| DeepPrivacy | 0.82 | 0.75 | **111.86** | 0.02 | 0.72 | 0.66 | **95.29** | 0.02 |
| FAMS | 0.76 | 0.70 | 804.29 | 0.01 | 0.68 | 0.64 | 548.03 | 0.01 |
| **Gauss-GDDiff** | 0.86 | 0.79 | 263.66 | 0.01 | 0.77 | 0.70 | 194.99 | 0.01 |
| CLEANIR | 0.79 | 0.71 | 303.70 | 0.01 | 0.69 | 0.65 | 449.87 | 0.01 |
| **Lpl-GDDiff** | 0.92 | 0.84 | 216.63 | 0.01 | 0.85 | 0.77 | 168.56 | 0.01 |
| CPP-DeID (0.9) | **0.60** | **0.57** | 2121.45 | 0.16 | **0.60** | **0.57** | 1615.58 | 0.17 |
| FADM | 0.90 | 0.82 | 332.70 | 0.01 | 0.82 | 0.75 | 230.59 | 0.02 |
| LDFA | 0.73 | 0.67 | 571.87 | 0.11 | 0.67 | 0.62 | 568.54 | 0.11 |
| CPP-DeID (0.5) | 0.92 | 0.85 | 1531.82 | 0.09 | 0.88 | 0.80 | 1057.98 | 0.10 |
| AMT-GAN | 0.95 | 0.91 | 820.16 | 0.04 | 0.93 | 0.87 | 710.13 | 0.04 |

Same pattern: CPP-DeID (0.9) buys its AUC of 0.60 with an FD of 2121.45 and MSE of 0.16.
Mean-GDDiff (0.5) reaches AUC 0.70 / 0.62 at FD 362.03 / 250.10 and MSE 0.03 — an order of magnitude
less distortion for comparable identity suppression.

### Overall ranking (Friedman test)

Across all four datasets and all metrics (AUC, F1, FD, MSE, gACC), the 13 methods differ
significantly: **χ²(12) = 57.75, p < 0.001**, Kendall's **W = 0.283** (modest agreement in rankings
across datasets, i.e. method performance is dataset-dependent).

| Method | Mean rank ↓ |
|:---|---:|
| **Mean-GDDiff (0.1)** | **4.73** |
| **Mean-GDDiff (0.2)** | **4.88** |
| **Mean-GDDiff (0.5)** | **5.20** |
| DeepPrivacy | 6.08 |
| FAMS | 6.11 |
| **Gauss-GDDiff** | 6.17 |
| CLEANIR | 6.56 |
| **Lpl-GDDiff** | 6.64 |
| CPP-DeID (0.9) | 7.17 |
| FADM | 7.20 |
| LDFA | 7.79 |
| CPP-DeID (0.5) | 11.20 |
| AMT-GAN | 11.20 |

**The three Mean-GDDiff variants take the top three mean ranks.** Under the Bonferroni–Dunn test
(CD = 3.827, α = 0.05), Mean-GDDiff (0.1) significantly outperforms LDFA, CPP-DeID (0.5), and
AMT-GAN; differences against the mid-field (DeepPrivacy, FADM, FAMS) are not statistically
significant.

### Data utility: landmarks and segmentation

Beyond gACC, utility is measured by comparing original vs. de-identified images on:

- **Procrustes distance (↓)** over 478 3D MediaPipe Face Mesh landmarks, after removing translation,
  scale, and rotation (SVD-optimal alignment) — geometric structure preservation;
- **Mean IoU (↑)** over BiSeNet face segmentation across 18 facial classes — structural region
  preservation.

The expected trade-off holds across methods: weaker de-identification (higher IV) buys better
segmentation and landmark consistency. GDDiff sits in the balanced middle of that frontier. It also
**preserves the original facial orientation**, whereas CPP-DeID biases RaFD outputs toward frontal
views, and it keeps gender attributes more consistently than FAMS (gACC 0.80–0.82 vs. 0.40 on RaFD).

### Ablation: fixed-variance and zero-kernel configurations (RaFD)

| Configuration | AUC ↓ | F1 ↓ | FD ↓ | MSE ↓ | gACC ↑ |
|:---|---:|---:|---:|---:|---:|
| Gauss Max | 0.92 | 0.84 | 644.07 | 0.009 | 0.81 |
| Gauss Min | 0.92 | 0.84 | 645.96 | 0.009 | 0.81 |
| Lpl Max | 0.96 | 0.90 | 571.32 | 0.007 | 0.81 |
| Lpl Min | 0.96 | 0.90 | 572.23 | 0.007 | 0.81 |
| Zero (zero kernel throughout) | **0.78** | **0.71** | 767.27 | 0.022 | 0.78 |
| **Mean-GDDiff (λ = 0.5)** | 0.77 | 0.70 | 747.91 | 0.02 | 0.80 |

`Min` / `Max` denote a fixed minimum / maximum variance held throughout diffusion. Two findings:

1. **Fixed variance barely matters** — Gauss Max vs. Gauss Min differ by 1.89 in FD and nothing
   elsewhere. The method is insensitive to the specific fixed variance.
2. **The zero kernel alone is not enough.** Cancelling all in-face guidance gives strong suppression
   (AUC 0.78) but the worst image quality and utility in the table (FD 767.27, MSE 0.022,
   gACC 0.78). Structured Gaussian / Lpl weighting is what buys back realism: full Mean-GDDiff
   (λ = 0.5) matches its suppression (AUC 0.77) at better FD, MSE, **and** gACC.

---

## Evaluation protocol

Identity verification metrics (details in the paper's supplementary Sec. 7) are all derived from
pairwise cosine similarity over AdaFace embeddings:

- **FAR(τ)** — fraction of non-mated pairs with score ≥ τ.
- **FRR(τ)** — fraction of mated pairs with score < τ.
- **EER** — the operating point where `FAR(τ*) = FRR(τ*)`. The reported **F1 uses the EER
  threshold**.
- **AUC** — `∫₀¹ TPR(FPR⁻¹(x)) dx`, with `TPR = 1 − FRR`, `FPR = FAR`. It measures the separability
  of mated from non-mated scores, so **lower AUC means stronger privacy protection**.

Supporting tooling in this repo:

- [AdaFace/](AdaFace/) — the face recognition model used for identity embeddings.
- [verif_AUC.py](verif_AUC.py) — bootstrapped FAR / FRR / EER / AUC with uncertainty estimates from
  score CSVs (columns `cossim`, `ground_truth`).
- [work_binary_benchmarks.ipynb](work_binary_benchmarks.ipynb),
  [verification_results.ipynb](verification_results.ipynb) — score analysis and distribution /
  overlap plots.
- [align_dataset.py](align_dataset.py) — face alignment following the CPP-DeID protocol.
- [equalize_ilumination.py](equalize_ilumination.py), [create_figure.py](create_figure.py) —
  preprocessing and figure helpers.

---

## Reproducibility notes

A few things to know before treating a local run as a reproduction of the paper's tables:

- **Reverse diffusion steps.** The paper reports **50** reverse diffusion steps. The shipped configs
  set `T_sampling: 100` ([configs/celeba_hq.yml](configs/celeba_hq.yml)). Set it to 50 to match the
  paper.
- **Hyperparameters.** The paper states η = 0.85, c = 1.0, ζ = 0.3. The shipped run scripts use
  `--eta_tilde 0.7 --scale_ls 1.0 --zeta 0.5`. Note that `eta_tilde` is **not** η directly — the
  code uses `η_reg = max(1e-4, σ_y² · eta_tilde)`, a different parameterization — so these values
  are not comparable term by term.
- **Measurement operator.** The paper's derivation specializes to `A = I` (pure denoising). The
  shipped scripts run `--deg deblur_gauss`, a mild near-uniform 5-tap blur
  ([diffusion.py:379-389](guided_diffusion/diffusion.py#L379-L389)). A true `--deg denoising` path
  also exists (SVD operators only).
- **Face detector.** The paper describes BlazeFace via MediaPipe for bbox alignment; this code uses
  the OpenCV Haar cascade plus a centered fallback box. MediaPipe Face Mesh is used only for the
  landmark-based utility evaluation.
- **Resolution.** The base diffusion model operates at 256×256. The paper downsamples 1024×1024
  inputs with bilinear interpolation and upsamples the de-identified outputs back with Lanczos
  resampling; that resize step is external to this sampler.
- **Determinism.** `--seed` (default `1234`) seeds torch, numpy, and the dataloader workers, but the
  dataloader uses `shuffle=True`, so per-image output filenames depend on the seed.

---

## Limitations and ethical considerations

- GDDiff makes **no formal privacy guarantee**. It does not assume a specific adversary and offers
  no differential-privacy bound; it empirically reduces identity-related embedding similarity while
  balancing image quality and task-relevant attributes.
- Identity suppression is measured against **AdaFace**, one representative state-of-the-art verifier.
  Different recognition architectures encode identity differently, so effectiveness against other
  verifiers may differ. Broader verifier ensembles are future work.
- Effectiveness **depends on accurate face detection**, and the 256×256 processing resolution may
  affect performance against higher-resolution verifiers.
- De-identification strength may **vary across demographic groups**, and the recognition models used
  to evaluate it carry their own demographic biases. Fairness-aware evaluation and fairness-aware
  guidance are open directions.
- Face de-identification is **dual-use**: it protects individuals from unauthorized recognition, but
  it can also be used to evade legitimate security systems. We advocate responsible development and
  deployment, with ongoing attention to fairness and transparency.

Future work: video anonymization, fairness-aware guidance, and evaluation against a wider ensemble
of recognition models.

---

## Data and code availability

All evaluation data come from publicly available datasets: **RaFD**, **XM2VTS**, **LFW**, and
**CALFW**. De-identified images produced by the method are available upon reasonable request. The
reference release of this framework will be published at
<https://github.com/ManfredGonzalez/GDDiff> upon acceptance.

---

GDDiff builds on the preconditioned-guidance sampler of DDPG, which should be cited alongside it:

```bibtex
@inproceedings{garber2024image,
  title     = {Image Restoration by Denoising Diffusion Models with
               Iteratively Preconditioned Guidance},
  author    = {Garber, Tomer and Tirer, Tom},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  year      = {2024}
}
```

---

## Funding and acknowledgments

Supported in part by the Slovenian Research Agency (ARIS) through Research Programmes P2-0250 (B)
"Metrology and Biometric Systems" and P2-0214 (A) "Computer Vision", the ARIS Project J2-50065
"DeepFake DAD", and the ARIS Young Researcher Programme; and by the European Union's Horizon Europe
research and innovation programme through the **OnMoveID** project (Grant Agreement No. 101225635).

We thank the creators of the RaFD, XM2VTS, LFW, and CALFW datasets, acknowledge the use of the
AdaFace face recognition model for identity evaluation, and gratefully acknowledge NVIDIA
Corporation for the donation of GPU hardware through the NVIDIA Academic GPU Grant Program.

---

## Credits

This implementation extends [tirer-lab/DDPG](https://github.com/tirer-lab/DDPG) (Garber & Tirer,
CVPR 2024), which is itself inspired by [DDRM](https://github.com/bahjat-kawar/ddrm) and
[DDNM](https://github.com/wyhuai/DDNM). The original DDPG code handles image restoration
(super-resolution, Gaussian / motion deblurring); GDDiff repurposes its preconditioned guidance for
spatially controlled face de-identification.
