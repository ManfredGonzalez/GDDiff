"""Approximate FLOPs of GDDiff per input image (reviewer R2.m2).

Cost is dominated by the denoiser: one UNet forward per branch per sampling step.
GDDiff runs 2 branches (G, L) until the fusion step and 3 (G, L, A) afterwards; the
fusion step is taken from the sampler's own schedule (get_schedule_jump,
get_percentile_index with per = 1 - (lambda + 0.1), as in guided_diffusion/diffusion.py).
At the fusion step itself the UNet still runs on 2 branches (A is appended after the
forward pass), so A's forwards start one step later.

UNet FLOPs are counted with torch.utils.flop_counter (convolutions, linear layers and
attention matmuls; multiply-add = 2 FLOPs). The pointwise guidance and the 5-tap blur
operator are not counted; they are negligible next to the UNet.

Usage (env: gddiff):  python analysis/flops_gddiff.py
"""
import os
import sys

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from functions.ddpg_scheme import get_percentile_index, get_schedule_jump  # noqa: E402
from guided_diffusion.models import Model  # noqa: E402


class Cfg(dict):
    __getattr__ = dict.__getitem__


def to_cfg(d):
    return Cfg({k: to_cfg(v) if isinstance(v, dict) else v for k, v in d.items()})


def main():
    config = to_cfg(yaml.safe_load(open('configs/celeba_hq.yml')))
    model = Model(config).eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, config.data.image_size, config.data.image_size)
    t = torch.ones(1) * 500
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x, t)
    f = counter.get_total_flops()
    T = config.sampling.T_sampling
    times = get_schedule_jump(T, 1, 1)
    pairs = list(zip(times[:-1], times[1:]))
    print(f'UNet: {n_params / 1e6:.1f} M parameters, {f / 1e9:.1f} GFLOPs per forward pass '
          f'(256x256, batch 1); sampling steps: {len(pairs)}')
    print(f'{"setting":34s} {"UNet forwards":>14s} {"TFLOPs/image":>13s} {"TFLOPs per output":>18s}')
    single = len(pairs)
    print(f'{"single-branch sampler (1 output)":34s} {single:14d} {single * f / 1e12:13.2f} {single * f / 1e12:18.2f}')
    print(f'{"G + L only, no average branch (2)":34s} {2 * single:14d} {2 * single * f / 1e12:13.2f} '
          f'{single * f / 1e12:18.2f}')
    for lam in (0.1, 0.2, 0.5):
        idx = get_percentile_index(pairs, percentile=100 * (1 - (lam + 0.1)))
        fwd = 2 * (idx + 2) + 3 * (len(pairs) - idx - 2)  # A joins the forward pass from step idx + 2
        print(f'{"GDDiff lambda=" + str(lam) + " (3 outputs)":34s} {fwd:14d} {fwd * f / 1e12:13.2f} '
              f'{fwd * f / 3e12:18.2f}   (A spawned after step {idx + 1}/{len(pairs)})')


if __name__ == '__main__':
    main()
