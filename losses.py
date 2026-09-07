"""Loss functions for MBBN.

Band-repulsion loss
-------------------
The published objective is ``-log(S)`` with
``S = L1(h,l) + L1(h,u) + L1(l,u)``, where each ``L1`` is a *mean* absolute
difference between three row-stochastic (softmax) attention maps. Two
properties of that choice matter and are the reason for the alternatives below
(see ``docs/audit/AUDIT.md`` for the measurements):

1. ``S`` is bounded above by ``6/N`` for ``N`` ROIs, and the bound is attained
   *only* when every attention row is one-hot and the three bands point at
   different targets. Minimising ``-log S`` therefore has its global optimum at
   single-target ("star"/hub) attention, independently of the data.
2. ``S -> 0`` whenever the three maps agree — which is exactly the state at
   initialisation — so ``-log S -> +inf`` and ``d/dA (-log S) ~ 1/S`` diverges.
   With ``spatial_loss_factor`` up to 100 this is the documented NaN source.

``minus_log`` is kept verbatim so published runs remain reproducible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask_Loss(nn.Module):
    """Reconstruction loss for masked pretraining.

    ``masked_only=True`` restricts the loss to positions the model could not
    see, which is the standard masked-modelling objective. The published
    behaviour (loss over every position, so the identity map on visible
    positions dominates) is ``masked_only=False``.
    """

    def __init__(self, **kwargs):
        super(Mask_Loss, self).__init__()
        self.masked_only = bool(kwargs.get('mask_loss_on_masked_only', False))
        self.criterion = nn.L1Loss(reduction='none')

    def forward(self, input_seq, output_seq, mask=None):
        # input_seq / output_seq : (batch, time, ROI)
        # mask                   : (batch, time, ROI) bool, True where hidden
        err = self.criterion(output_seq, input_seq)
        if not self.masked_only or mask is None:
            return err.mean()
        mask = mask.to(err.dtype)
        denom = mask.sum()
        if denom == 0:
            return err.mean()
        return (err * mask).sum() / denom


class Spatial_Difference_Loss(nn.Module):
    """Encourage the three band-specific spatial attention maps to differ.

    ``loss_type``
        ``minus_log``     published: ``-log(S)``. Unbounded, singular at ``S=0``.
        ``minus_log_eps`` ``-log(S + eps)``; same shape, finite everywhere.
        ``neg_linear``    ``1 - S / S_max`` with ``S_max = 6/N``; in ``[0, 1]``,
                          gradient magnitude independent of ``S``.
        ``cosine``        mean pairwise cosine similarity of row-centred maps.
                          Invariant to how peaked a row is, so it separates the
                          bands without rewarding one-hot attention.

    ``entropy_weight`` adds ``-w * H(rows)``-style pressure back toward
    non-degenerate attention; use it with ``minus_log``/``neg_linear`` if you
    want repulsion without hub collapse.
    """

    _TYPES = ('minus_log', 'minus_log_eps', 'neg_linear', 'cosine')

    def __init__(self, **kwargs):
        super(Spatial_Difference_Loss, self).__init__()
        self.loss_type = kwargs.get('spat_diff_loss_type', 'minus_log')
        if self.loss_type not in self._TYPES:
            raise ValueError(f'spat_diff_loss_type must be one of {self._TYPES}, got {self.loss_type!r}')
        self.eps = float(kwargs.get('spat_diff_eps', 1e-4))
        self.entropy_weight = float(kwargs.get('spat_diff_entropy_weight', 0.0))
        self.l1 = nn.L1Loss()

    @staticmethod
    def _row_entropy(m):
        p = m.reshape(-1, m.shape[-1]).clamp_min(1e-12)
        return -(p * p.log()).sum(-1).mean()

    @staticmethod
    def _pairwise_cosine(a, b):
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean()

    def forward(self, h, l, u):
        # h, l, u : (batch, heads, ROI, ROI) row-stochastic attention maps
        if self.loss_type == 'cosine':
            loss = (self._pairwise_cosine(h, l)
                    + self._pairwise_cosine(h, u)
                    + self._pairwise_cosine(l, u)) / 3.0
        else:
            S = self.l1(h, l) + self.l1(h, u) + self.l1(l, u)
            if self.loss_type == 'minus_log':
                loss = -torch.log(S)                      # published, verbatim
            elif self.loss_type == 'minus_log_eps':
                loss = -torch.log(S + self.eps)
            else:                                          # neg_linear
                s_max = 6.0 / h.shape[-1]
                loss = 1.0 - S / s_max

        if self.entropy_weight > 0:
            h_max = torch.log(torch.tensor(float(h.shape[-1]), device=h.device))
            ent = (self._row_entropy(h) + self._row_entropy(l) + self._row_entropy(u)) / 3.0
            loss = loss + self.entropy_weight * (h_max - ent) / h_max
        return loss
