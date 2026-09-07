"""Architecture regressions: head, gradient wiring, padding, masking."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from model import (Attention, Classifier, Transformer_Block, pick_spatial_heads,
                   Transformer_Finetune_Three_Channels,
                   Transformer_Reconstruction_Three_Channels)

Transformer_Block.init_weights = lambda self, *a, **k: None   # keep tests fast


@pytest.mark.parametrize('seq,expect_divides', [(280, True), (348, True), (464, True),
                                                (96, True), (347, True)])
def test_pick_spatial_heads_always_divides(seq, expect_divides):
    """Released code tried 12 then 8 with no else, raising UnboundLocalError."""
    h = pick_spatial_heads(seq)
    assert seq % h == 0 and h >= 1


def test_attention_can_return_features():
    """`return_attn=False` hit `self.proj`, which was never created."""
    a = Attention(dim=24, num_heads=4)
    out = a(torch.randn(2, 8, 24), return_attn=False)
    assert out.shape == (2, 8, 24)


def test_attn_only_drops_the_unused_value_projection():
    full = sum(p.numel() for p in Attention(dim=24, num_heads=4).qkv.parameters())
    lean = sum(p.numel() for p in Attention(dim=24, num_heads=4, attn_only=True).qkv.parameters())
    assert lean == pytest.approx(full * 2 / 3, rel=1e-6)


def test_published_head_makes_a_prediction_depend_on_its_batch_mates():
    """BatchNorm1d on a scalar logit couples subjects within a batch."""
    torch.manual_seed(0)
    feats = torch.randn(256, 32)
    subj = feats[:1]

    def spread(head_type):
        torch.manual_seed(0)
        head = Classifier(32, 1, dropout=0.0, head_type=head_type).train()
        vals = [head(torch.cat([subj, feats[torch.randperm(255)[:15] + 1]]))[0, 0].item()
                for _ in range(80)]
        return float(np.std(vals))

    assert spread('published') > 0.05
    assert spread('linear') == pytest.approx(0.0, abs=1e-7)


def test_published_head_zeroes_most_training_logits():
    torch.manual_seed(0)
    head = Classifier(32, 1, dropout=0.6, head_type='published').train()
    z = head(torch.randn(2000, 32))
    assert 0.5 < (z == 0).float().mean().item() < 0.7


def _mbbn(**over):
    kw = dict(intermediate_vec=24, num_heads=4, transformer_hidden_layers=1,
              transformer_dropout_rate=0.0, spatiotemporal=True, gpu=False,
              dataset_name='ABIDE', fine_tune_task='binary_classification',
              target='ASD', step='2', visualization=False, finetune_test=False,
              pretrained_model_weights_path=None, finetune=False,
              temporal_masking_window_size=4, window_interval_rate=2,
              num_hub_ROIs=12, communicability_option='remove_high_comm_node')
    kw.update(over)
    return Transformer_Finetune_Three_Channels(sequence_length=48, **kw)


def test_spatial_maps_are_disconnected_from_the_prediction_by_default():
    """The headline interpretability finding, pinned as a test.

    In the released model the band-specific spatial attention receives gradient
    only from the band-repulsion loss, so the maps reported as disorder
    signatures carry no label gradient at all.
    """
    torch.manual_seed(0)
    m = _mbbn().eval()
    xs = [torch.randn(2, 48, 24) for _ in range(3)]
    m.zero_grad()
    m(*xs)['binary_classification'].sum().backward()
    total = sum(p.grad.abs().sum().item() for n, p in m.named_parameters()
                if 'spatial_attention' in n and p.grad is not None)
    assert total == 0.0


def test_spatial_head_restores_the_label_gradient():
    torch.manual_seed(0)
    m = _mbbn(spatial_head=True).eval()
    xs = [torch.randn(2, 48, 24) for _ in range(3)]
    m.zero_grad()
    m(*xs)['binary_classification'].sum().backward()
    total = sum(p.grad.abs().sum().item() for n, p in m.named_parameters()
                if 'spatial_attention' in n and p.grad is not None)
    assert total > 0.0


def test_band_embedding_distinguishes_identical_band_inputs():
    """Shared encoder + shared head: without band identity the 3 bands are one
    function evaluated 3 times."""
    torch.manual_seed(0)
    x = torch.randn(2, 48, 24)

    plain = _mbbn().eval()
    o = plain(x, x.clone(), x.clone())
    a = plain.transformer(x)['cls']
    assert torch.allclose(a, plain.transformer(x.clone())['cls'], atol=1e-6)

    band = _mbbn(band_embedding=True).eval()
    with torch.no_grad():
        band.band_embed.normal_(0, 0.5)
    c = [band.transformer(x, band_offset=band._band_offset(i))['cls'] for i in range(3)]
    assert not torch.allclose(c[0], c[1], atol=1e-4)
    assert not torch.allclose(c[0], c[2], atol=1e-4)


def test_padding_mask_excludes_padded_timepoints():
    """With attention_mask=None the encoder pools over pure-zero padding."""
    torch.manual_seed(0)
    m = _mbbn().eval()
    real = torch.randn(2, 30, 24)
    padded = torch.zeros(2, 48, 24)
    padded[:, 9:39] = real
    valid = torch.zeros(2, 48, dtype=torch.bool)
    valid[:, 9:39] = True

    with torch.no_grad():
        unmasked = m.transformer(padded)['cls']
        masked = m.transformer(padded, valid_mask=valid)['cls']
    assert not torch.allclose(unmasked, masked, atol=1e-5)


def _recon(comm_dir, **over):
    kw = dict(intermediate_vec=24, num_heads=4, transformer_hidden_layers=1,
              transformer_dropout_rate=0.0, spatiotemporal=True, gpu=False,
              dataset_name='ABIDE', fine_tune_task='binary_classification',
              target='reconstruction', step='3', visualization=False,
              finetune_test=False, pretrained_model_weights_path=None, finetune=False,
              temporal_masking_window_size=4, window_interval_rate=2,
              num_hub_ROIs=12, communicability_option='remove_high_comm_node',
              communicability_dir=comm_dir, communicability_dataset='ABIDE')
    kw.update(over)
    return Transformer_Reconstruction_Three_Channels(sequence_length=48, **kw)


def test_published_mask_is_identical_for_every_subject(synth_root):
    comm = str(synth_root / 'communicability')
    from fixtures.synthetic import make_communicability
    make_communicability(str(synth_root), n_roi=24, dataset='ABIDE')
    m = _recon(comm).eval()
    x = torch.randn(4, 48, 24)
    _, mask = m._apply_mask(x, 0)
    assert torch.equal(mask[0], mask[1]) and torch.equal(mask[0], mask[3])


def test_random_mask_differs_across_subjects_and_epochs(synth_root):
    from fixtures.synthetic import make_communicability
    make_communicability(str(synth_root), n_roi=24, dataset='ABIDE')
    comm = str(synth_root / 'communicability')
    m = _recon(comm, random_mask=True, mask_ratio_spatial=0.3,
               mask_ratio_temporal=0.3).eval()
    x = torch.randn(4, 48, 24)
    torch.manual_seed(0); _, m1 = m._apply_mask(x, 0)
    torch.manual_seed(1); _, m2 = m._apply_mask(x, 0)
    assert not torch.equal(m1[0], m1[1])
    assert not torch.equal(m1, m2)


def test_mask_token_is_learned_not_zero(synth_root):
    from fixtures.synthetic import make_communicability
    make_communicability(str(synth_root), n_roi=24, dataset='ABIDE')
    m = _recon(str(synth_root / 'communicability'), random_mask=True)
    x = torch.randn(2, 48, 24)
    out = m(x, x.clone(), x.clone())
    out['mask_spatiotemporal_high_fmri_sequence'].sum().backward()
    assert m.mask_token.grad is not None and m.mask_token.grad.abs().sum() > 0


def test_hub_order_loaded_once_at_construction(synth_root):
    from fixtures.synthetic import make_communicability
    make_communicability(str(synth_root), n_roi=24, dataset='ABIDE')
    m = _recon(str(synth_root / 'communicability'))
    assert m.hub_order.shape == (3, 24)
