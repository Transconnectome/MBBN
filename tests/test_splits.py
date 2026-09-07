"""Split construction: grouping, site stratification, leave-one-site-out."""
import numpy as np
import pytest

from data_preprocess_and_load.dataloaders import make_splits


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    n = 400
    ids = np.array([f's{i:04d}' for i in range(n)])
    groups = np.array([f'fam{i // 2}' for i in range(n)])       # sibling pairs
    sites = np.array([['A', 'B', 'C'][i % 3] for i in range(n)])
    y = (rng.random(n) < 0.4).astype(int)
    return ids, y, groups, sites


def test_family_grouping_keeps_relatives_in_one_fold(cohort):
    ids, y, groups, _ = cohort
    tr, va, te = make_splits(ids, y, seed=0, groups=groups)
    gmap = dict(zip(ids, groups))
    fold_of = {}
    for name, part in (('train', tr), ('val', va), ('test', te)):
        for s in part:
            fold_of.setdefault(gmap[s], set()).add(name)
    leaked = {g: f for g, f in fold_of.items() if len(f) > 1}
    assert not leaked, f'{len(leaked)} families split across folds'


def test_ungrouped_split_leaks_relatives(cohort):
    """Documents what the released split does with ABCD siblings."""
    ids, y, groups, _ = cohort
    tr, va, te = make_splits(ids, y, seed=0, stratify_labels=y)
    gmap = dict(zip(ids, groups))
    fold_of = {}
    for name, part in (('train', tr), ('val', va), ('test', te)):
        for s in part:
            fold_of.setdefault(gmap[s], set()).add(name)
    assert sum(len(f) > 1 for f in fold_of.values()) > 20


def test_site_stratification_balances_sites(cohort):
    ids, y, _, sites = cohort
    strat = np.array([f'{a}|{b}' for a, b in zip(y.astype(str), sites)])
    tr, va, te = make_splits(ids, y, seed=0, stratify_labels=strat)
    smap = dict(zip(ids, sites))
    frac = lambda part: np.array([np.mean([smap[s] == k for s in part]) for k in 'ABC'])
    assert np.abs(frac(tr) - frac(te)).max() < 0.08


def test_leave_one_site_out_isolates_the_site(cohort):
    ids, y, _, sites = cohort
    strat = np.array([f'{a}|{b}' for a, b in zip(y.astype(str), sites)])
    mask = sites == 'B'
    tr, va, te = make_splits(ids, y, seed=0, stratify_labels=strat,
                             leave_out_mask=mask)
    smap = dict(zip(ids, sites))
    assert set(te) == set(ids[mask])
    assert all(smap[s] != 'B' for s in list(tr) + list(va))


def test_all_partitions_are_disjoint_and_complete(cohort):
    ids, y, groups, sites = cohort
    for kw in ({}, {'groups': groups}, {'stratify_labels': y}):
        tr, va, te = make_splits(ids, y, seed=1, **kw)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(ids)


def test_split_seed_separates_partition_from_init(cohort):
    ids, y, _, _ = cohort
    a = make_splits(ids, y, seed=7, stratify_labels=y)
    b = make_splits(ids, y, seed=7, stratify_labels=y)
    c = make_splits(ids, y, seed=8, stratify_labels=y)
    assert set(a[2]) == set(b[2])
    assert set(a[2]) != set(c[2])


def test_split_file_roundtrip_preserves_the_last_subject(tmp_path):
    """`[x[:-1] for x in lines]` truncated the final id."""
    ids = ['s001', 's002', 's003']
    p = tmp_path / 'split.txt'
    body = 'train_subjects\n' + '\n'.join(ids[:1]) + '\nval_subjects\n' + ids[1] + \
           '\ntest_subjects\n' + ids[2] + '\n'
    p.write_text(body)
    lines = [x.rstrip('\n') for x in open(p).readlines() if x.strip()]
    assert lines[-1] == 's003'
    released = [x[:-1] for x in open(p).readlines()]
    assert released[-1] == 's003'    # only survives because we now write a trailing newline


def test_site_labels_are_keyed_by_subject_id():
    """Regression: sites must never be read out of a frame positionally.

    `new_meta` is shorter than and reordered relative to `metadata`, so zipping
    a positional site column against the subject list pairs subjects with other
    rows' sites (and `zip` truncates instead of raising).
    """
    import pandas as pd
    from data_preprocess_and_load.dataloaders import DataHandler

    metadata = pd.DataFrame({
        'SUB_ID': [10, 11, 12, 13, 14, 15],
        'DX_GROUP': [1, 2, 1, 2, 1, 2],
        'SITE_ID': ['A', 'B', 'C', 'A', 'B', 'C'],
    })
    # only 3 subjects have imaging, and in a different order
    new_meta = metadata[metadata['SUB_ID'].isin([14, 10, 12])][['SUB_ID', 'DX_GROUP']]
    new_meta = new_meta.iloc[::-1]
    ids = new_meta['SUB_ID'].values

    sites = DataHandler._sites_for(None, ids, 'SUB_ID', new_meta, metadata, None)
    truth = dict(zip(metadata['SUB_ID'].astype(str), metadata['SITE_ID']))
    assert list(sites) == [truth[str(i)] for i in ids]
    assert len(sites) == len(ids)


def test_site_lookup_falls_back_to_the_pheno_file(tmp_path):
    import pandas as pd
    from data_preprocess_and_load.dataloaders import DataHandler

    metadata = pd.DataFrame({'SUB_ID': [10, 11, 12], 'DX_GROUP': [1, 2, 1]})
    new_meta = metadata[['SUB_ID', 'DX_GROUP']]
    aux = tmp_path / 'pheno.csv'
    pd.DataFrame({'SUB_ID': [12, 10, 11], 'SITE_ID': ['C', 'A', 'B']}).to_csv(aux, index=False)

    sites = DataHandler._sites_for(None, new_meta['SUB_ID'].values, 'SUB_ID',
                                   new_meta, metadata, str(aux))
    assert list(sites) == ['A', 'B', 'C']


def test_site_lookup_raises_when_no_site_is_available():
    import pandas as pd
    import pytest as _pytest
    from data_preprocess_and_load.dataloaders import DataHandler
    md = pd.DataFrame({'SUB_ID': [1, 2], 'DX_GROUP': [1, 2]})
    with _pytest.raises(KeyError):
        DataHandler._sites_for(None, md['SUB_ID'].values, 'SUB_ID', md, md, None)


def test_joint_stratification_degrades_when_the_fold_is_too_small():
    """ABIDE has 17 sites: 34 strata cannot fit a 15% test fold of ~21 subjects."""
    import numpy as np
    import pytest as _pytest
    rng = np.random.default_rng(0)
    n = 140
    ids = np.array([f's{i:04d}' for i in range(n)])
    y = (rng.random(n) < 0.45).astype(int)
    sites = np.array([f'SITE{i % 17}' for i in range(n)])
    strat = np.array([f'{a}|{b}' for a, b in zip(y.astype(str), sites)])
    with _pytest.warns(RuntimeWarning, match='falling back to target-only'):
        tr, va, te = make_splits(ids, y, seed=0, stratify_labels=strat)
    assert len(tr) + len(va) + len(te) == n
    ymap = dict(zip(ids, y))
    # the marginal (diagnosis) balance is still preserved
    assert abs(np.mean([ymap[s] for s in te]) - y.mean()) < 0.15
