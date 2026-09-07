"""Argument routing, weight loading, reproducibility."""
import os
import re
import warnings
import pytest


def test_sort_args_warns_when_an_option_belongs_to_another_phase():
    """This is how pretrain_MBBN.slurm silently trained at 348 instead of 464."""
    from utils import sort_args
    args = {'sequence_length_phase3': 464, 'sequence_length_phase4': 348, 'seed': 1}
    with pytest.warns(RuntimeWarning, match='NO effect on step 3'):
        out = sort_args('3', args, explicit={'sequence_length_phase4'})
    assert out['sequence_length'] == 464


def test_sort_args_silent_when_nothing_is_dropped():
    from utils import sort_args
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = sort_args('3', {'sequence_length_phase3': 464, 'seed': 1},
                        explicit={'sequence_length_phase3'})
    assert out['sequence_length'] == 464


def test_weight_loader_raises_on_a_missing_path(tmp_path):
    """Released code swallowed every failure in a bare `except:`."""
    from utils import weight_loader
    import argparse
    args = argparse.Namespace(step='2', model_weights_path_phase2=str(tmp_path / 'nope.pth'),
                              pretrained_model_weights_path=None)
    with pytest.raises(FileNotFoundError):
        weight_loader(args)


def test_weight_loader_returns_none_when_no_path_given():
    from utils import weight_loader
    import argparse
    args = argparse.Namespace(step='2', model_weights_path_phase2=None,
                              pretrained_model_weights_path=None)
    path, step, task = weight_loader(args)
    assert path is None and step == '2' and task == 'MBBN'


def test_phase_sequence_length_defaults_are_phase_appropriate():
    src = open(os.path.join(os.path.dirname(__file__), '..', 'main.py')).read()
    p3 = re.search(r"--sequence_length_phase3', type=int, default=(\d+)", src)
    p4 = re.search(r"--sequence_length_phase4', type=int, default=(\d+)", src)
    assert p3 and p4, 'both phase-tagged sequence lengths must be declared'
    assert int(p3.group(1)) == 464, 'phase 3 (pretraining on UKB) is 464 timepoints'


def test_pretraining_script_uses_a_phase3_flag():
    p = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'main_experiments',
                     '02_pretraining', 'pretrain_MBBN.slurm')
    live = [l for l in open(p).read().splitlines()
            if not l.lstrip().startswith('#')]
    assert not any('--sequence_length_phase4' in l for l in live), \
        'a phase-4 flag in the pretraining script is discarded by sort_args'
    assert any('--sequence_length_phase3' in l for l in live)
