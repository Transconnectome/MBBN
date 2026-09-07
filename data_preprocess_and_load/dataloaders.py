import numpy as np
import torch
from torch.utils.data import DataLoader,Subset, Dataset, RandomSampler, SequentialSampler
import random
from pathlib import Path
#DDP
from torch.utils.data.distributed import DistributedSampler
from data_preprocess_and_load.datasets import *
from utils import reproducibility
import os
import nibabel as nib
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
import pandas as pd
import warnings

SITE_COLUMN_CANDIDATES = ('SITE_ID', 'site', 'SITE', 'site_id', 'abcd_site', 'site_name')
GROUP_COLUMN_CANDIDATES = ('rel_family_id', 'rel_family_id_bl', 'family_id', 'FAMILY_ID',
                           'famid', 'rel_relationship_id')


def canon_id(x):
    """Canonical subject key for joining metadata frames.

    The same subject appears in three different spellings across this codebase:
    the imaging directory is `0050001`, the ABIDE metadata carries the integer
    `50001`, and the ABIDE split branch re-pads to the string `'0050001'`. A
    dict keyed on any one spelling silently misses the others, so normalise
    (strip whitespace, drop a trailing `.0` from float-parsed ids, strip
    leading zeros) before building or querying any id -> attribute mapping.
    """
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s.lstrip('0') or '0'


def resolve_column(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f'could not find a {what} column in the metadata; looked for {candidates}. '
        f'Available columns: {sorted(df.columns)[:40]}')


def make_splits(ids, y, seed, stratify_labels=None, groups=None,
                test_size=0.15, val_size=0.175, leave_out_mask=None):
    """Partition subject ids into train / val / test.

    Three additions over the released `train_test_split(stratify=y)`:

    * `groups` -- keeps a family (ABCD siblings and twins) inside one fold.
      Splitting related subjects across folds lets the model recognise a
      relative rather than generalise.
    * `stratify_labels` -- stratify on target x site jointly. ABIDE pools 17
      sites with different scanners, protocols and diagnostic base rates, so a
      target-only split leaves the site distribution free to differ between
      folds and site becomes a usable shortcut.
    * `leave_out_mask` -- assign an entire site to the test fold, which is the
      only version of this evaluation that answers "does it transfer to a
      scanner it has never seen".
    """
    ids = np.asarray(ids)
    y = np.asarray(y)

    if leave_out_mask is not None:
        leave_out_mask = np.asarray(leave_out_mask, dtype=bool)
        if leave_out_mask.sum() == 0:
            raise ValueError('leave_one_site_out matched no subjects')
        test_ids = ids[leave_out_mask]
        rest = ~leave_out_mask
        sub_strat = None if stratify_labels is None else np.asarray(stratify_labels)[rest]
        sub_groups = None if groups is None else np.asarray(groups)[rest]
        tr, va, _ = make_splits(ids[rest], y[rest], seed, sub_strat, sub_groups,
                                test_size=val_size, val_size=0.0)
        return list(tr), list(va), list(test_ids)

    if groups is not None and stratify_labels is not None:
        # stratified *and* grouped
        n_splits = max(2, int(round(1.0 / test_size)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr_idx, te_idx = next(sgkf.split(ids, stratify_labels, groups))
        if val_size <= 0:
            return list(ids[tr_idx]), list(ids[te_idx]), []
        n_inner = max(2, int(round(1.0 / val_size)))
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(ids[tr_idx],
                                      np.asarray(stratify_labels)[tr_idx],
                                      np.asarray(groups)[tr_idx]))
        return (list(ids[tr_idx][i_tr]), list(ids[tr_idx][i_va]), list(ids[te_idx]))

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(ids, y, groups))
        if val_size <= 0:
            return list(ids[tr_idx]), list(ids[te_idx]), []
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        i_tr, i_va = next(inner.split(ids[tr_idx], y[tr_idx], np.asarray(groups)[tr_idx]))
        return (list(ids[tr_idx][i_tr]), list(ids[tr_idx][i_va]), list(ids[te_idx]))

    strat = stratify_labels
    if strat is not None:
        strat = np.asarray(strat)
        # StratifiedShuffleSplit needs >= 2 members per stratum
        vals, counts = np.unique(strat, return_counts=True)
        rare = set(vals[counts < 4])
        if rare:
            warnings.warn(f'{len(rare)} stratum/strata have <4 subjects and were '
                          f'collapsed for splitting: {sorted(rare)[:10]}', RuntimeWarning)
            strat = np.array(['__RARE__' if s in rare else s for s in strat])

        # A fold cannot contain fewer subjects than there are strata. Joint
        # target x site stratification is simply infeasible on a small cohort:
        # ABIDE has 17 sites, so 34 strata do not fit a 15% test fold of ~21
        # subjects. Degrade to the marginal (target-only) stratification and say
        # so, rather than crashing or silently splitting on nothing.
        n_strata = len(np.unique(strat))
        n_test = int(np.floor(test_size * len(ids)))
        if n_strata > max(n_test, 1):
            marginal = np.array([str(s).split('|')[0] for s in strat])
            warnings.warn(
                f'joint stratification needs {n_strata} strata but the test fold '
                f'holds only {n_test} subjects; falling back to target-only '
                f'stratification ({len(np.unique(marginal))} strata). Site cannot '
                f'be balanced across folds at this cohort size -- consider '
                f'leave_one_site_out or k-fold instead.', RuntimeWarning)
            strat = marginal

    X_tr, X_te = train_test_split(ids, test_size=test_size, stratify=strat,
                                  random_state=seed)
    if val_size <= 0:
        return list(X_tr), list(X_te), []
    strat_tr = None
    if strat is not None:
        pos = {v: i for i, v in enumerate(ids)}
        strat_tr = np.array([strat[pos[v]] for v in X_tr])
    X_tr2, X_va = train_test_split(X_tr, test_size=val_size, stratify=strat_tr,
                                   random_state=seed)
    return list(X_tr2), list(X_va), list(X_te)


class DataHandler():
    def __init__(self,test=False,**kwargs):
        self.step = kwargs.get('step')
        self.base_path = kwargs.get('base_path')
        self.kwargs = kwargs
        self.dataset_name = kwargs.get('dataset_name')
        self.target = kwargs.get('target')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.seq_len = kwargs.get('sequence_length')
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.seed = kwargs.get('seed')
        # `--split_seed` decouples the partition from weight init, so repeated
        # "seeds" can vary one at a time instead of both together
        self.split_seed = kwargs.get('split_seed')
        if self.split_seed is None:
            self.split_seed = self.seed
        self.site_stratify = bool(kwargs.get('site_stratify', False))
        self.group_by_family = bool(kwargs.get('group_by_family', False))
        self.leave_one_site_out = kwargs.get('leave_one_site_out')
        reproducibility(**self.kwargs)
        dataset = self.get_dataset()
        self.train_dataset = dataset(**self.kwargs)
        self.eval_dataset = dataset(**self.kwargs)
        if self.fine_tune_task == 'regression':
            if self.step != '3':
                self.mean = self.train_dataset.mean
                self.std = self.train_dataset.std
            elif self.step == '3':
                self.mean = self.eval_dataset.mean
                self.std = self.eval_dataset.std
        self.eval_dataset.augment = None

        if self.target == 'ADHD_label':
            self.target = 'ADHD'
        elif self.target == 'ASD':
            # Released code wrote `self.target == 'DX_GROUP'` here -- a comparison
            # whose result is discarded, so this branch is a no-op. That turns out
            # to be the correct behaviour and is kept: the ABIDE branch below
            # renames the DX_GROUP column to 'ASD' before splitting, and for ABCD
            # the target is mapped back to the metadata's own 'ASD_label' column
            # further down, so the frame passed to _split always carries a column
            # that resolves. Left as an
            # explicit no-op rather than "fixed" into an assignment, which would
            # only change the split filename and invalidate existing split files.
            pass
        elif self.target == 'nihtbx_totalcomp_uncorrected':
            self.target = 'total_intelligence'
        elif self.target == 'ASD_label':
            if self.dataset_name == 'ABCD':
                self.target = 'ASD'
        
        self.splits_folder = Path(self.base_path).joinpath('splits',self.dataset_name)
        suffix = ''
        if self.site_stratify:
            suffix += '_siteStrat'
        if self.group_by_family:
            suffix += '_famGroup'
        if self.leave_one_site_out:
            suffix += f'_LOSO-{self.leave_one_site_out}'
        self.current_split = self.splits_folder.joinpath(f"{self.dataset_name}_{self.target}_ROI_{self.intermediate_vec}_seq_len_{self.seq_len}_split{self.split_seed}{suffix}.txt")
        
        if not self.current_split.exists():
            print('generating splits...')
            # go back to origianl target in metadata
  
            if self.target == 'ADHD':
                self.target = 'ADHD_label'
            elif self.target == 'total_intelligence':
                self.target = 'nihtbx_totalcomp_uncorrected'
            elif self.target == 'ASD':
                if self.dataset_name == 'ABCD':
                    self.target = 'ASD_label' 
            elif self.target == 'depression':
                self.target = 'MDD_pp'
            
            ## generate stratified sampler
            if self.dataset_name == 'ABCD':
                sub = [i.split('-')[1] for i in os.listdir(kwargs.get('abcd_path'))]
                if self.target == 'MDD_pp':
                    metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABCD_5_1_KSADS_raw_MDD_ANX_CorP_pp_pres_ALL.csv'))
                    metadata['subjectkey'] = [i.split('-')[1] for i in metadata['subjectkey']]
                else: 
                    metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABCD_matched.csv'))
                    # metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABCD_phenotype_total.csv'))
            elif self.dataset_name == 'UKB':
                sub = [str(i).split('_20227')[0] for i in os.listdir(kwargs.get('ukb_path'))]
                metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/UKB_phenotype_gps_fluidint.csv'))
            elif self.dataset_name == 'ABIDE':
                sub = os.listdir(kwargs.get('abide_path'))
                metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABIDE1+2_meta.csv'))
                
            if self.target == 'SuicideIdeationtoAttempt':
                new_meta = metadata[['subjectkey', 'sex', self.target]].dropna()
                        
            elif self.target == 'reconstruction':
                new_meta = None

            else:
                if self.dataset_name == 'ABCD':
                    new_meta = metadata[['subjectkey', self.target]].dropna()
                elif self.dataset_name == 'UKB':
                    new_meta = metadata[['eid', self.target]].dropna()
                    new_meta['eid'] = new_meta['eid'].astype('object')
                elif self.dataset_name == 'ABIDE':
                    new_meta = metadata[['SUB_ID', 'DX_GROUP']].dropna()

            # 01 remove subjects which has NaN voxel from the original 4D data
            print('generating step 1')
            valid_sub = []
            prob_sub = []
            
            for i in sub:
                if self.dataset_name == 'ABCD':
                    if self.intermediate_vec == 360:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'hcp_mmp1_sub-'+i+'.npy')
                    elif self.intermediate_vec == 400:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'schaefer_sub-'+i+'.npy')
                    file = np.load(filename)[:self.seq_len].T
                elif self.dataset_name == 'UKB':
                    if self.intermediate_vec == 360:
                        j = i+'_20227_2_0'
                        k = i+'_20227_3_0'
                        
                        filename_j = os.path.join(kwargs.get('ukb_path'), j+'/'+'hcp_mmp1_360_'+j+'.npy')
                        filename_k = os.path.join(kwargs.get('ukb_path'), k+'/'+'hcp_mmp1_360_'+k+'.npy')

                        if os.path.exists(filename_j):
                            filename = filename_j
                            subname = j
                        elif os.path.exists(filename_k):
                            filename = filename_k
                            subname = k
                        
                    elif self.intermediate_vec == 400:
                        filename = os.path.join(kwargs.get('ukb_path'), i+'/'+'schaefer_400Parcels_17Networks_'+i+'.npy')
                    file = np.load(filename)[20:20+self.seq_len].T
                
                elif self.dataset_name == 'ABIDE':
                    if self.intermediate_vec == 400:
                        filename = os.path.join(kwargs.get('abide_path'), i+'/'+'schaefer_400Parcels_17Networks_'+i+'.npy')
                    elif self.intermediate_vec == 360:
                        filename = os.path.join(kwargs.get('abide_path'), i+'/'+'hcp_mmp1_360_'+i+'.npy')
                
                    file = np.load(filename)[:self.seq_len].T
                    
                if file.shape[1] >= self.seq_len:
                    valid_sub.append(i)
                    for j in range(file.shape[0]):
                        if np.std(file[j]) == 0:
                            ## Constant ROI: band-pass filtering + z-scoring would
                            ## turn it into NaN, so drop the subject.
                            prob_sub.append(i)
                            break
            valid_sub = list(set(valid_sub) - set(prob_sub))
                
            if self.dataset_name == 'UKB':
                valid_sub = list(map(int, valid_sub))
            print('valid sub', len(valid_sub))
                        
            # 02 select subjects with target and split file
            print('generating step 2')
            if self.target == 'reconstruction':
                sublist = ['train_subjects']+list(valid_sub)+['val_subjects']+['test_subjects']+[' ']
            else:
                if self.dataset_name == 'ABCD':
                    valid_df = pd.DataFrame(valid_sub).rename(columns = {0 : 'subjectkey'})
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='subjectkey')

                    if self.target == 'SuicideIdeationtoAttempt':
                        '''stratified sampling for two columns'''
                        X_train, X_test = train_test_split(new_meta['subjectkey'],
                                          test_size=0.15,
                                          stratify= new_meta[['sex', self.target]],
                                          random_state = self.seed)

                        train_and_valid = new_meta[new_meta['subjectkey'].isin(X_train)]

                        X_train, X_valid = train_test_split(train_and_valid['subjectkey'],
                                                          test_size=0.175,
                                                          stratify= train_and_valid[['sex', self.target]],
                                                          random_state = self.seed)
                    else:
                        X_train, X_valid, X_test = self._split(new_meta, 'subjectkey', metadata)
                            
                            
                elif self.dataset_name == 'UKB':
                    valid_df = pd.DataFrame(valid_sub).rename(columns = {0 : 'eid'})
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='eid')
                    
                    #eid_map = {s.split('_')[0]: s for s in valid_sub_fullname}
                    #new_meta['eid'] = new_meta['eid'].astype(int).astype(str).map(eid_map).fillna(new_meta['eid'])
                    
                    X_train, X_valid, X_test = self._split(new_meta, 'eid', metadata)
                elif self.dataset_name == 'ABIDE':
                    
                    subid = valid_sub # [i[2:] if i.startswith('00') else i for i in valid_sub]
                    # starts with 5 or 2 now! (because metadata doesn't starts with 00)
                    valid_df = pd.DataFrame(subid).rename(columns = {0 : 'SUB_ID'}) # ['SUB_ID']
                    new_meta = new_meta.rename(columns={'DX_GROUP': 'ASD'})  # both dataframes use int IDs

                    new_meta['SUB_ID'] = new_meta['SUB_ID'].astype(str)
                    new_meta['SUB_ID'] = ['00' + i if '00' + i in subid else i for i in new_meta['SUB_ID']]
                    # now new_meta's SUB_ID contains 00
                    
                    valid_df['SUB_ID'] = valid_df['SUB_ID'].astype(str)
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='SUB_ID')
                    
                    if self.target == 'DX_GROUP':
                        self.target = 'ASD'
                    X_train, X_valid, X_test = self._split(new_meta, 'SUB_ID', metadata,
                                                           site_source=self.site_meta_path())
                    
                
                
                sublist = ['train_subjects']+list(X_train)+['val_subjects']+list(X_valid)+['test_subjects']+list(X_test)
                if self.target == 'ADHD_label':
                    self.target = 'ADHD'
                elif self.target == 'nihtbx_totalcomp_uncorrected':
                    self.target = 'total_intelligence'
                elif self.target == 'ASD_label':
                    self.target = 'ASD'
                elif self.target == 'MDD_pp':
                    self.target = 'depression'
            
            if self.dataset_name == 'UKB':
                sublist = list(map(str, sublist))
                
            print('generating step 3.. saving splits...')
            # write to the same path that is read back; the released code rebuilt
            # the filename by hand, so any option that changes it silently wrote
            # one file and read another
            self.current_split.parent.mkdir(parents=True, exist_ok=True)
            with open(self.current_split, mode="w") as file:
                file.write('\n'.join(str(s) for s in sublist) + '\n')
        print(self.current_split)

    def site_meta_path(self):
        return os.path.join(self.base_path, 'data', 'metadata', 'ABIDE1_pheno_and_sites.csv')

    def _sites_for(self, ids, id_col, new_meta, metadata, site_source=None):
        """Acquisition site per entry of `ids`, looked up BY SUBJECT ID.

        `new_meta` is `metadata[[id_col, target]].dropna()` intersected with the
        subjects that actually have imaging, so it is both shorter than
        `metadata` and in a different order. Reading a site column positionally
        out of either frame and zipping it against `ids` therefore pairs
        subjects with unrelated rows' sites -- and `zip` truncates silently
        rather than raising. Always go through an id -> site mapping.
        """
        candidates = [df for df in (new_meta, metadata) if df is not None]
        if site_source and os.path.exists(site_source):
            candidates.append(pd.read_csv(site_source))

        want = [canon_id(i) for i in ids]
        for df in candidates:
            try:
                site_col = resolve_column(df, SITE_COLUMN_CANDIDATES, 'site')
            except KeyError:
                continue
            try:
                key = resolve_column(df, (id_col, 'SUB_ID', 'subjectkey', 'eid'), 'id')
            except KeyError:
                continue
            mapping = {canon_id(k): str(v) for k, v in zip(df[key], df[site_col])}
            sites = np.array([mapping.get(s, 'UNKNOWN') for s in want])
            n_missing = int((sites == 'UNKNOWN').sum())
            if n_missing == len(sites):
                continue                       # this frame keys on something else
            if n_missing:
                warnings.warn(f'{n_missing}/{len(sites)} subjects have no site in '
                              f'column {site_col!r}; labelled UNKNOWN', RuntimeWarning)
            print(f'site labels from column {site_col!r} keyed on {key!r}: '
                  f'{len(set(sites))} distinct sites')
            return sites
        raise KeyError(
            'site stratification / leave_one_site_out was requested but no frame '
            f'provided both a site column {SITE_COLUMN_CANDIDATES} and a usable '
            f'subject-id column. Checked new_meta, metadata and {site_source!r}.')

    def _split(self, new_meta, id_col, metadata, site_source=None):
        """Dispatch to make_splits() with whatever grouping was requested."""
        ids = new_meta[id_col].values
        # Take the label column from the frame rather than from self.target: the
        # branches above rename it per dataset (ABIDE selects 'DX_GROUP' then
        # renames it to 'ASD'; the suicide target carries an extra 'sex' column),
        # so self.target and the frame's column name can legitimately differ.
        label_col = [c for c in new_meta.columns if c != id_col][-1]
        y = new_meta[label_col].values

        strat = None
        if self.site_stratify or self.leave_one_site_out:
            sites = self._sites_for(ids, id_col, new_meta, metadata, site_source)
            strat = np.array([f'{a}|{b}' for a, b in zip(np.asarray(y).astype(str), sites)])
        elif self.fine_tune_task == 'binary_classification':
            strat = np.asarray(y)

        groups = None
        if self.group_by_family:
            gcol = resolve_column(metadata, GROUP_COLUMN_CANDIDATES, 'family/group')
            idc = resolve_column(metadata, (id_col,), 'id')
            gmap = {canon_id(k): str(v) for k, v in zip(metadata[idc], metadata[gcol])}
            groups = np.array([gmap.get(canon_id(i), f'__solo_{canon_id(i)}') for i in ids])
            n_solo = int(sum(1 for g in groups if g.startswith('__solo_')))
            if n_solo:
                warnings.warn(f'{n_solo}/{len(groups)} subjects had no family id and were '
                              f'treated as singletons', RuntimeWarning)

        leave_out = None
        if self.leave_one_site_out:
            leave_out = np.array([self.leave_one_site_out in s.split('|')[-1] for s in strat])

        tr, va, te = make_splits(ids, y, self.split_seed, stratify_labels=strat,
                                 groups=groups, leave_out_mask=leave_out)
        print(f'split sizes  train={len(tr)}  val={len(va)}  test={len(te)}'
              f'  (site_stratify={self.site_stratify}, group_by_family={self.group_by_family},'
              f' LOSO={self.leave_one_site_out})')
        return tr, va, te
        
    def get_mean_std(self):
        return None
    
    def get_dataset(self):
        if self.dataset_name == 'ABCD':
            return ABCD_fMRI_timeseries
        elif self.dataset_name == 'ABIDE':
            return ABIDE_fMRI_timeseries
        elif self.dataset_name == 'UKB':
            return UKB_fMRI_timeseries
        
    def current_split_exists(self):
        return self.current_split.exists() 


    def create_dataloaders(self):
        reproducibility(**self.kwargs) 

        # `x[:-1]` strips the last character of every line, which for the final
        # line (no trailing newline) removed a real character from the last test
        # subject id, silently dropping that subject from evaluation.
        self.subject_list = self.train_dataset.index_l
        
        if not self.current_split_exists():
            raise FileNotFoundError(f'split file missing: {self.current_split}')
        print('loading splits')
        train_names, val_names, test_names = self.load_split()
        train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(train_names,val_names,test_names,self.subject_list)
        for nm, idx, names in (('train', train_idx, train_names), ('val', val_idx, val_names),
                               ('test', test_idx, test_names)):
            if names and len(idx) != len(names):
                warnings.warn(f'{nm}: split file lists {len(names)} subjects but only '
                              f'{len(idx)} matched the dataset index', RuntimeWarning)
        

        print('length of train_idx:', len(train_idx))
        print('length of val_idx:', len(val_idx))
        print('length of test_idx:', len(test_idx))
        
        train_dataset = Subset(self.train_dataset, train_idx)
        val_dataset = Subset(self.eval_dataset, val_idx)
        test_dataset = Subset(self.eval_dataset, test_idx)
        
        if self.kwargs.get('distributed'):
            print('distributed')
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            print('length of train sampler is:', len(train_sampler)) # 22
            if self.target != 'reconstruction':
                # NOTE: DistributedSampler pads the last rank's shard by repeating
                # subjects, so metrics aggregated across ranks double-count them.
                # Evaluate on one rank, or use drop_last=False + dedup by subject.
                valid_sampler = DistributedSampler(val_dataset, shuffle=False)
                print('length of valid sampler is:', len(valid_sampler)) # 5
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
                print('length of test sampler is:', len(test_sampler))
        else:
            train_sampler = RandomSampler(train_dataset)
            if self.target != 'reconstruction':
                valid_sampler = SequentialSampler(val_dataset)
                test_sampler = SequentialSampler(test_dataset)
        
        ## Stella transformed this part ##
        training_generator = DataLoader(train_dataset, **self.get_params(**self.kwargs),
                                       sampler=train_sampler)
        print('length of training generator is:', len(training_generator))
        
        if self.target != 'reconstruction':
            val_generator = DataLoader(val_dataset, **self.get_params(eval=True,**self.kwargs),
                                      sampler=valid_sampler)
            print('length of valid generator is:', len(val_generator))
            
            test_generator = DataLoader(test_dataset, **self.get_params(eval=True,**self.kwargs),
                               sampler=test_sampler)
            print('length of test generator is:', len(test_generator))
        
        
        else:
            val_generator = None
            test_generator = None
           
                   
        if self.fine_tune_task == 'regression':
            return training_generator, val_generator, test_generator, self.mean, self.std
            
        else:
            return training_generator, val_generator, test_generator
    
    
    
    def get_params(self,eval=False,**kwargs):
        batch_size = kwargs.get('batch_size')
        workers = kwargs.get('workers')
        cuda = kwargs.get('cuda')
        seed = kwargs.get('seed', 0)

        def worker_init_fn(worker_id):
            s = (seed * 10007 + worker_id) % (2 ** 31 - 1)
            np.random.seed(s)
            random.seed(s)

        params = {'batch_size': batch_size,
                  'num_workers': min(8, workers),
                  # never drop subjects from an evaluation fold
                  'drop_last': False if eval else True,
                  'pin_memory': False,
                  'persistent_workers': False,
                  'prefetch_factor': 1 if workers > 0 else None,
                  'worker_init_fn': worker_init_fn}
        return params

    def convert_subject_list_to_idx_list(self,train_names,val_names,test_names,subj_list):
        subj_idx = np.array([str(x[1]) for x in subj_list])
        # np.in1d was removed in NumPy 2.0; np.isin is the supported spelling
        train_idx = np.where(np.isin(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.isin(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.isin(subj_idx, test_names))[0].tolist()
        
        return train_idx,val_idx,test_idx
    
    def load_split(self):
        subject_order = open(self.current_split, 'r').readlines()
        subject_order = [x.rstrip('\n') for x in subject_order]
        subject_order = [x for x in subject_order if x != '']
        train_index = np.argmax(['train' in line for line in subject_order])
        val_index = np.argmax(['val' in line for line in subject_order])
        test_index = np.argmax(['test' in line for line in subject_order])
        train_names = subject_order[train_index + 1:val_index]
        val_names = subject_order[val_index+1:test_index]
        test_names = subject_order[test_index + 1:]
              
        return train_names,val_names,test_names
