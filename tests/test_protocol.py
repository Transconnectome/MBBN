"""Evaluation-protocol regressions: operating point, dropped subjects."""
import warnings
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from metrics import Metrics


@pytest.fixture(scope='module')
def scored():
    rng = np.random.default_rng(0)
    truth = (rng.random(400) < 0.3).astype(int)
    pred = np.clip(0.5 + 0.25 * (truth - 0.5) + rng.normal(0, 0.12, 400), 1e-4, 1 - 1e-4)
    return truth, pred


def test_threshold_zero_makes_test_metrics_degenerate(scored):
    """Released path: Writer's val_threshold (0) is what test thresholds against."""
    truth, pred = scored
    met = Metrics()
    with pytest.warns(RuntimeWarning, match='outside'):
        acc, thr, gm, spec, sens, f1 = met.ROC_CURVE(truth, list(pred), 'test', 0)
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(0.0)
    assert acc == pytest.approx(0.5)
    p = truth.mean()
    assert f1 == pytest.approx(2 * p / (1 + p), rel=1e-6)
    assert roc_auc_score(truth, pred) > 0.9      # the model itself is fine


def test_carrying_the_val_threshold_recovers_real_metrics(scored):
    truth, pred = scored
    met = Metrics()
    v_acc, v_thr, *_ = met.ROC_CURVE(truth, list(pred), 'val', 0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        t_acc, t_thr, t_gm, t_spec, t_sens, t_f1 = met.ROC_CURVE(truth, list(pred), 'test', v_thr)
    assert t_thr == pytest.approx(v_thr)
    assert t_acc > 0.8 and t_spec > 0.5 and t_sens > 0.5


def test_writer_carries_val_threshold_to_test():
    import inspect
    import loss_writer
    src = inspect.getsource(loss_writer.Writer.accuracy_summary)
    assert 'self.val_threshold = metrics' in src, \
        'Writer must update val_threshold from the validation ROC'


def test_eval_loaders_never_drop_subjects():
    import inspect
    from data_preprocess_and_load.dataloaders import DataHandler
    src = inspect.getsource(DataHandler.get_params)
    assert "'drop_last': False if eval else True" in src


def test_drop_last_would_discard_subjects():
    """Quantifies what the released eval loaders silently excluded."""
    for n, b, expect in [(150, 32, 22), (300, 16, 12), (600, 32, 24)]:
        assert n % b == expect


def test_spatial_loss_warmup_only_applies_while_training():
    """An eval-only Trainer starts at step 0; the ramp must not zero the loss."""
    import inspect
    import trainer as tr
    src = inspect.getsource(tr.Trainer.aggregate_losses)
    assert "getattr(self, 'mode', 'train') == 'train'" in src, \
        'the spatial-loss warmup must be gated on training mode'
