"""Evaluation-protocol regressions: operating point, dropped subjects."""
import inspect
import pathlib
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


def test_nvtx_helpers_delegate_and_do_not_recurse(monkeypatch):
    """Guarded CUDA helpers must be exercisable without a GPU.

    On a CPU wheel `torch.cuda.is_available()` is False, so a bug in the
    guarded branch is invisible locally and only surfaces on a GPU box.
    Force the branch and assert it delegates to torch.cuda.nvtx exactly once.
    """
    import torch
    import trainer as tr
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, 'range_push', lambda tag: calls.append(('push', tag)))
    monkeypatch.setattr(torch.cuda.nvtx, 'range_pop', lambda: calls.append(('pop',)))
    tr._nvtx_push('probe')
    tr._nvtx_pop()
    assert calls == [('push', 'probe'), ('pop',)]


def test_every_cuda_only_call_sits_behind_a_guard():
    """A CUDA-only call on a CPU wheel is fatal, and CPU CI cannot see it.

    Static check over trainer.py: every torch.cuda call that requires a CUDA
    build must live in a function that also tests torch.cuda.is_available().
    """
    import ast
    import trainer as tr
    CUDA_ONLY = {'empty_cache', 'reset_peak_memory_stats', 'max_memory_allocated',
                 'max_memory_reserved', 'max_memory_cached', 'memory_cached',
                 'synchronize', 'range_push', 'range_pop'}

    def attr_path(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr); node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))

    src = pathlib.Path(tr.__file__).read_text()
    tree = ast.parse(src)
    unguarded = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        guarded = 'is_available' in (ast.get_source_segment(src, fn) or '')
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                path = attr_path(n.func)
                if path.startswith('torch.cuda.') and path.split('.')[-1] in CUDA_ONLY and not guarded:
                    unguarded.append((fn.name, n.lineno, path))
    assert not unguarded, f'unguarded CUDA-only calls: {unguarded}'


def test_nvtx_helper_does_not_call_itself():
    """Regression: a regex sweep once rewrote the helper's own body,
    so _nvtx_push recursed until RecursionError -- invisible on CPU because
    the is_available() guard short-circuits, fatal on a GPU box."""
    import trainer as tr
    for fn in (tr._nvtx_push, tr._nvtx_pop):
        body = inspect.getsource(fn)
        assert 'torch.cuda.nvtx.' in body, f'{fn.__name__} must delegate to torch.cuda.nvtx'
        assert body.count(fn.__name__) == 1, f'{fn.__name__} calls itself'
