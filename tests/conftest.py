import os
import sys
import pathlib
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tests'))

from fixtures.synthetic import make_abide, make_communicability   # noqa: E402


@pytest.fixture(scope='session')
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('mbbn_synth')
    make_abide(str(root))
    make_communicability(str(root))
    return root


@pytest.fixture(scope='session')
def model_kwargs():
    """Small MBBN configuration that still uses the real ROI/atlas dimensions."""
    return dict(
        intermediate_vec=24, num_heads=4, transformer_hidden_layers=2,
        transformer_dropout_rate=0.1, spatiotemporal=True, gpu=False,
        dataset_name='ABIDE', fine_tune_task='binary_classification',
        target='ASD', step='2', visualization=False, finetune_test=False,
        pretrained_model_weights_path=None, finetune=False,
        temporal_masking_window_size=4, window_interval_rate=2,
        num_hub_ROIs=12, communicability_option='remove_high_comm_node',
        sequence_length=48,
    )
