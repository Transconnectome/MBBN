"""End-to-end: the full step-2 pipeline on synthetic data, on CPU.

The released repo cannot be executed without ABCD/UKB/ABIDE, a W&B account and
a GPU, so nothing about it was testable. This runs `main.py` to completion on
generated ROI time series and asserts the pieces the audit fixed actually work:
a checkpoint is written, the selected-checkpoint path exists, and the held-out
fold is scored once with the validation operating point.
"""
import json
import os
import subprocess
import sys
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

COMMON = [
    '--dataset_name', 'ABIDE', '--fine_tune_task', 'binary_classification',
    '--target', 'ASD', '--intermediate_vec', '360', '--num_heads', '12',
    '--transformer_hidden_layers', '2', '--transformer_dropout_rate', '0.1',
    '--filtering_type', 'Boxcar', '--wandb_mode', 'disabled', '--seed', '1',
]


def _run(args, cwd, extra_env=None):
    env = dict(os.environ, PYTHONPATH=str(REPO), WANDB_MODE='disabled',
               TOKENIZERS_PARALLELISM='false', **(extra_env or {}))
    return subprocess.run([sys.executable, str(REPO / 'main.py')] + args,
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=2400)


@pytest.fixture(scope='module')
def trained(synth_root, tmp_path_factory):
    work = tmp_path_factory.mktemp('run')
    args = COMMON + [
        '--step', '2',
        '--base_path', str(synth_root),
        '--abide_path', str(synth_root / 'ABIDE_ROI'),
        '--sequence_length_phase2', '96',
        '--nEpochs_phase2', '2', '--batch_size_phase2', '4',
        '--lr_init_phase2', '1e-4', '--workers_phase2', '0',
        '--spatial_loss_factor', '1.0',
        '--exp_name', 'e2e',
        # hardened defaults under test
        '--head_type', 'linear', '--spat_diff_loss_type', 'neg_linear',
        '--spatial_loss_warmup', '5', '--spatial_head', '--band_embedding',
        '--site_stratify', '--cache_bands', '--nan_policy', 'raise',
    ]
    r = _run(args, cwd=str(work))
    return r, synth_root


def test_pipeline_runs_to_completion(trained):
    r, _ = trained
    assert r.returncode == 0, f'main.py failed\nSTDOUT tail:\n{r.stdout[-4000:]}\nSTDERR tail:\n{r.stderr[-4000:]}'


def test_selected_checkpoint_exists(trained):
    """run_phase() used to return a filename save_checkpoint_ never writes."""
    r, _ = trained
    line = [l for l in r.stdout.splitlines() if l.startswith('selected checkpoint:')]
    assert line, f'no checkpoint was selected\n{r.stdout[-3000:]}'
    path = line[-1].split('selected checkpoint:', 1)[1].strip()
    assert os.path.exists(path), f'{path} does not exist'


def test_held_out_fold_is_scored_once_with_the_val_threshold(trained):
    r, _ = trained
    assert 'Evaluating the selected checkpoint on the held-out test fold' in r.stdout
    used = [l for l in r.stdout.splitlines() if 'using loaded threshold' in l]
    assert used, 'test evaluation never reported its operating point'
    thr = float(used[-1].split('-')[1].strip())
    assert 0.0 < thr < 1.0, f'test threshold {thr} is degenerate (val threshold not carried)'


def test_band_cache_is_populated(trained):
    r, root = trained
    cache = pathlib.Path(root) / 'cache' / 'bands'
    assert cache.exists() and list(cache.glob('*.npz')), 'band cache was never written'


def test_no_nan_losses(trained):
    r, _ = trained
    assert 'found nans in computation' not in r.stdout


def test_split_file_written_where_it_is_read(trained):
    r, root = trained
    files = list((pathlib.Path(root) / 'splits' / 'ABIDE').glob('*siteStrat*.txt'))
    assert files, 'site-stratified split file was not written to the path read back'
    body = files[0].read_text().splitlines()
    for header in ('train_subjects', 'val_subjects', 'test_subjects'):
        assert header in body
