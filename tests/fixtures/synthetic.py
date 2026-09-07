"""Synthetic ROI-timeseries cohorts, so the pipeline runs with no real data.

The generator produces the on-disk layout the ABIDE loader expects
(`<root>/ABIDE_ROI/00<id>/hcp_mmp1_360_00<id>.npy` plus the two metadata CSVs)
with AR(1) parcel time series and a weak class-dependent coherent component, so
a short run has something learnable without any real subject data.
"""
import os
import numpy as np
import pandas as pd

SITES = {'NYU': 2.0, 'PITT': 1.5, 'USM': 2.0}


def _ar1_timeseries(n_roi, seq_len, rng, phi=0.85, scale=20.0):
    x = np.zeros((n_roi, seq_len))
    e = rng.standard_normal((n_roi, seq_len))
    for t in range(1, seq_len):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return scale * x


def make_abide(root, n_subjects=48, n_roi=360, seq_len=96, seed=0,
               effect=0.9, n_signal_roi=12):
    rng = np.random.default_rng(seed)
    data_dir = os.path.join(root, 'ABIDE_ROI')
    meta_dir = os.path.join(root, 'data', 'metadata')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    site_names = list(SITES)
    rows, site_rows = [], []
    signal_roi = rng.choice(n_roi, size=n_signal_roi, replace=False)

    for i in range(n_subjects):
        sub_id = 50001 + i
        folder = f'00{sub_id}'
        os.makedirs(os.path.join(data_dir, folder), exist_ok=True)

        dx = 1 if i % 2 == 0 else 2            # 1 = ASD, 2 = control (ABIDE coding)
        site = site_names[i % len(site_names)]
        y = _ar1_timeseries(n_roi, seq_len, rng)

        # class-dependent shared oscillation in a fixed ROI subset
        t = np.arange(seq_len)
        f = 0.05 if dx == 1 else 0.02
        common = np.sin(2 * np.pi * f * t * SITES[site])
        y[signal_roi] += effect * y[signal_roi].std() * common

        np.save(os.path.join(data_dir, folder, f'hcp_mmp1_360_{folder}.npy'),
                y.T.astype(np.float32))            # on disk as [time, ROI]
        rows.append({'SUB_ID': sub_id, 'SEX': 1 + (i % 2), 'DX_GROUP': dx})
        site_rows.append({'SUB_ID': sub_id, 'SITE_ID': site})

    pd.DataFrame(rows).to_csv(os.path.join(meta_dir, 'ABIDE1+2_meta.csv'), index=False)
    pd.DataFrame(site_rows).to_csv(os.path.join(meta_dir, 'ABIDE1_pheno_and_sites.csv'),
                                   index=False)
    return data_dir


def make_communicability(root, n_roi=360, dataset='ABIDE', seed=0):
    """Fake hub ordering so step-3 masking is exercisable."""
    rng = np.random.default_rng(seed)
    out = os.path.join(root, 'communicability')
    os.makedirs(out, exist_ok=True)
    for band in ('high', 'low', 'ultralow'):
        np.save(os.path.join(out, f'{dataset}_new_{band}_comm_ROI_order_ROI{n_roi}.npy'),
                rng.permutation(n_roi))
    return out
