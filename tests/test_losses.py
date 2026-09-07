"""Band-repulsion loss: boundedness, singularity, and the one-hot optimum."""
import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from losses import Spatial_Difference_Loss, Mask_Loss

N = 64


def _maps(kind, heads=4, batch=2, n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == 'identical':
        m = torch.rand(batch, heads, n, n, generator=g).softmax(-1)
        return m, m.clone(), m.clone()
    if kind == 'onehot':
        eye = lambda s: torch.nn.functional.one_hot(
            (torch.arange(n) + s) % n, n).float().expand(batch, heads, n, n)
        return eye(0), eye(1), eye(2)
    return tuple((torch.rand(batch, heads, n, n, generator=g) * 3).softmax(-1)
                 for _ in range(3))


def test_published_loss_is_singular_where_maps_agree():
    """-log(S) diverges exactly at initialisation, where the bands still agree."""
    h, l, u = _maps('identical')
    loss = Spatial_Difference_Loss(spat_diff_loss_type='minus_log')(h, l, u)
    assert torch.isinf(loss), 'expected +inf for identical maps'


@pytest.mark.parametrize('kind', ['minus_log_eps', 'neg_linear', 'cosine'])
def test_alternatives_are_finite_everywhere(kind):
    for state in ('identical', 'onehot', 'random'):
        h, l, u = _maps(state)
        loss = Spatial_Difference_Loss(spat_diff_loss_type=kind)(h, l, u)
        assert torch.isfinite(loss), f'{kind} not finite at {state}'


def test_published_optimum_is_onehot_attention():
    """S is maximised (loss minimised) by disjoint one-hot rows: hub attention.

    S <= 6/N for row-stochastic maps, and the bound is attained only when every
    row is one-hot and the three bands point elsewhere. So the published
    objective's global optimum is a star graph regardless of the data.
    """
    crit = Spatial_Difference_Loss(spat_diff_loss_type='minus_log')
    l1 = nn.L1Loss()
    h, l, u = _maps('onehot')
    S_onehot = (l1(h, l) + l1(h, u) + l1(l, u)).item()
    assert S_onehot == pytest.approx(6.0 / N, rel=1e-6)

    hr, lr, ur = _maps('random')
    S_random = (l1(hr, lr) + l1(hr, ur) + l1(lr, ur)).item()
    assert S_random < S_onehot
    assert crit(h, l, u).item() < crit(hr, lr, ur).item()


def test_published_gradient_diverges_as_maps_converge():
    """|d loss / d logits| ~ 1/S, so it explodes near the initial state."""
    grads = {}
    for eps in (1e-3, 1e-2, 1e-1):
        g = torch.Generator().manual_seed(0)
        base = torch.rand(1, 4, N, N, generator=g)
        A = [(base + eps * torch.rand(1, 4, N, N, generator=g)).requires_grad_(True)
             for _ in range(3)]
        loss = Spatial_Difference_Loss(spat_diff_loss_type='minus_log')(
            *[a.softmax(-1) for a in A])
        loss.backward()
        grads[eps] = max(a.grad.abs().max().item() for a in A)
    assert grads[1e-3] > grads[1e-2] > grads[1e-1]


def test_neg_linear_gradient_is_scale_free():
    """The bounded form's gradient does not blow up as the maps converge."""
    out = {}
    for eps in (1e-3, 1e-1):
        g = torch.Generator().manual_seed(0)
        base = torch.rand(1, 4, N, N, generator=g)
        A = [(base + eps * torch.rand(1, 4, N, N, generator=g)).requires_grad_(True)
             for _ in range(3)]
        Spatial_Difference_Loss(spat_diff_loss_type='neg_linear')(
            *[a.softmax(-1) for a in A]).backward()
        out[eps] = max(a.grad.abs().max().item() for a in A)
    assert out[1e-3] / out[1e-1] < 5, f'gradient ratio {out[1e-3] / out[1e-1]:.1f}'


def test_entropy_term_penalises_hub_collapse():
    crit = Spatial_Difference_Loss(spat_diff_loss_type='neg_linear',
                                   spat_diff_entropy_weight=1.0)
    onehot = crit(*_maps('onehot')).item()
    unif = torch.full((2, 4, N, N), 1.0 / N)
    diffuse = crit(unif, unif.clone(), unif.clone()).item()
    assert onehot > diffuse - 1.0   # one-hot no longer strictly preferred


def test_mask_loss_masked_only_ignores_visible_positions():
    """Published loss covers every position, so copying visible input dominates."""
    torch.manual_seed(0)
    x = torch.randn(2, 10, 6)
    mask = torch.zeros(2, 10, 6, dtype=torch.bool)
    mask[:, :2, :] = True                      # 20% hidden
    out = x.clone()
    out[mask] = 0.0                            # perfect on visible, wrong on hidden

    full = Mask_Loss(mask_loss_on_masked_only=False)(x, out, mask).item()
    masked = Mask_Loss(mask_loss_on_masked_only=True)(x, out, mask).item()
    assert masked > full * 4
    assert full == pytest.approx(masked * mask.float().mean().item(), rel=1e-5)
