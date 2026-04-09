import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from iminuit import Minuit
from numba import njit
import warnings

warnings.filterwarnings("ignore")

from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer


@njit
def lorentzian_function(x, s0, f1):
    """Lorentzian power spectral density model (Eq. 2 in paper)."""
    return (s0 * f1**2) / (x**2 + f1**2)


@njit
def spline_multifractal(x, beta_low, beta_high, A, f2, smoothness):
    """Spline multifractal model with cubic-spline transition (Eq. 3-4 in paper)."""
    log_x = np.log(x)
    log_f2 = np.log(f2)
    w = np.where(log_x < log_f2 - smoothness, 0,
         np.where(log_x > log_f2 + smoothness, 1,
         0.5 * (1 - np.cos(np.pi * (log_x - log_f2 + smoothness) / (2 * smoothness)))))
    return A * x**(beta_low * (1 - w) + beta_high * w)


def _find_knee_frequencies(y, TR, seq_len, intermediate_vec,
                            lz_p0, lz_bounds):
    """Identify f1 (ultralow/low) and f2 (low/high) knee frequencies.

    Step 1: Lorentzian fit on the full PSD → f1
    Step 2: Spline multifractal fit on PSD above f1 → f2

    Returns (f1, f2) in Hz.
    """
    # Average over ROIs to get a single representative power spectrum
    sample_whole = y.mean(axis=0)

    T = TimeSeries(sample_whole, sampling_interval=TR)
    S = SpectralAnalyzer(T)
    xdata = np.array(S.spectrum_fourier[0][1:])
    ydata = np.abs(S.spectrum_fourier[1][1:])

    # Step 1 — Lorentzian: find f1
    popt, _ = curve_fit(lorentzian_function, xdata, ydata,
                        p0=lz_p0, bounds=lz_bounds, maxfev=50000)
    f1 = popt[-1]
    knee = round(f1 / (1 / (seq_len * TR)))
    if knee <= 0:
        knee = 1

    # Step 2 — Spline multifractal: find f2
    def least_squares(beta_low, beta_high, A, f2, smoothness):
        y_pred = spline_multifractal(xdata[knee:], beta_low, beta_high,
                                     A, f2, smoothness)
        return np.sum((y_pred - ydata[knee:])**2)

    m = Minuit(least_squares, beta_low=-1.2, beta_high=-0.5,
               A=10, f2=0.08, smoothness=0.25)
    m.limits['beta_low'] = (-5, -0.01)
    m.limits['beta_high'] = (-5, -0.01)
    m.limits['f2'] = (f1 + 0.0001, 0.15)
    m.limits['A'] = (1, 30)
    m.limits['smoothness'] = (0.001, 1)
    m.migrad()
    f2 = m.values['f2']

    return f1, f2


def _filter_three_bands(y, TR, f1, f2, filtering_type):
    """Split timeseries y [ROI × time] into ultralow, low, and high bands.

    Uses f2 as the high-pass cutoff and f1 as the low-pass cutoff.
    FIR is used for ultralow/low; Boxcar for high (as in Table 7).
    """
    # High frequency (above f2)
    T1 = TimeSeries(y, sampling_interval=TR)
    FA1 = FilterAnalyzer(T1, lb=f2)
    if filtering_type == 'FIR':
        high = FA1.fir.data
        # Guard against zero-variance ROIs after FIR filtering
        std = np.std(high, axis=1, keepdims=True)
        high = (high - high.mean(axis=1, keepdims=True)) / (std + 1e-10)
        ultralow_low = FA1.data - FA1.fir.data
    else:  # Boxcar
        high = stats.zscore(FA1.filtered_boxcar.data, axis=1)
        ultralow_low = FA1.data - FA1.filtered_boxcar.data

    # Low and ultralow frequencies (split at f1)
    T2 = TimeSeries(ultralow_low, sampling_interval=TR)
    FA2 = FilterAnalyzer(T2, lb=f1)
    if filtering_type == 'FIR':
        low = stats.zscore(FA2.fir.data, axis=1)
        ultralow = stats.zscore(FA2.data - FA2.fir.data, axis=1)
    else:  # Boxcar
        low = stats.zscore(FA2.filtered_boxcar.data, axis=1)
        ultralow = stats.zscore(FA2.data - FA2.filtered_boxcar.data, axis=1)

    return ultralow, low, high


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def register_args(self, **kwargs):
        self.index_l = []
        self.target = kwargs.get('target')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.dataset_name = kwargs.get('dataset_name')
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.filtering_type = kwargs.get('filtering_type')
        self.sequence_length = kwargs.get('sequence_length')
        self.pretrained_model_weights_path = kwargs.get('pretrained_model_weights_path')
        self.finetune = kwargs.get('finetune')
        self.transfer_learning = bool(self.pretrained_model_weights_path) or self.finetune
        self.finetune_test = kwargs.get('finetune_test')


# ─── ABIDE ────────────────────────────────────────────────────────────────────

class ABIDE_fMRI_timeseries(BaseDataset):
    """ABIDE-I/II dataset. Target: ASD (binary classification)."""

    # TR values per acquisition site (in seconds)
    _SITE_TR = {
        'CALTECH': 2.0, 'CMU': 2.0, 'KKI': 2.5, 'LEUVEN': 1.66,
        'MAX_MUN': 3.0, 'NYU': 2.0, 'OHSU': 2.5, 'OLIN': 1.5,
        'PITT': 1.5, 'SBL': 2.2, 'SDSU': 2.0, 'STANFORD': 2.0,
        'TRINITY': 2.0, 'UCLA': 3.0, 'UM': 2.0, 'USM': 2.0, 'YALE': 2.0,
    }

    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('abide_path')
        self.meta_data = pd.read_csv(
            os.path.join(kwargs.get('base_path'), 'data', 'metadata',
                         'ABIDE1+2_meta.csv'))
        self.site_meta_data = pd.read_csv(
            os.path.join(kwargs.get('base_path'), 'data', 'metadata',
                         'ABIDE1_pheno_and_sites.csv'))

        # Build file list
        data_list = []
        for sub in os.listdir(self.data_dir):
            if self.intermediate_vec == 400:
                data_list.append(os.path.join(
                    self.data_dir, sub,
                    'schaefer_400Parcels_17Networks_' + sub + '.npy'))
            elif sub.startswith('00'):  # ABIDE-I subjects
                data_list.append(os.path.join(
                    self.data_dir, sub, 'hcp_mmp1_360_' + sub + '.npy'))

        if self.intermediate_vec == 360:
            unified_name_list = [i[2:] for i in os.listdir(self.data_dir)
                                 if i.startswith('00')]
        else:
            unified_name_list = [i[2:] if i.startswith('00') else i
                                 for i in os.listdir(self.data_dir)]

        non_na = self.meta_data.dropna(axis=0)
        valid_sub = (set(str(i) for i in non_na['SUB_ID'])
                     & set(unified_name_list))

        for filename in data_list:
            sub = filename.split('/')[-2]
            if sub.startswith('00'):  # ABIDE-I
                subid = sub[2:]
                site_row = self.site_meta_data[
                    self.site_meta_data['SUB_ID'] == int(subid)]['SITE_ID']
                site = site_row.values[0] if len(site_row) else 'ABIDE2'
            else:
                subid = sub
                site = 'ABIDE2'

            if subid in valid_sub:
                if self.target == 'sex':
                    target_val = non_na.loc[
                        non_na['SUB_ID'] == int(subid), 'SEX'].values[0]
                else:  # ASD
                    target_val = non_na.loc[
                        non_na['SUB_ID'] == int(subid), 'DX_GROUP'].values[0]
                target = torch.tensor(1.0 if target_val == 2 else 0.0)
                self.index_l.append((subid, sub, filename, target, site))

    def __len__(self):
        return len(self.index_l)

    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target, site = self.index_l[index]

        y = np.load(path_to_fMRIs)[:self.sequence_length].T  # [ROI, seq_len]

        # Padding to pretrained sequence length (464) for finetuning
        pad = 464 - self.sequence_length

        # Site-specific TR
        TR = next((v for k, v in self._SITE_TR.items() if k in site), 3.0)

        f1, f2 = _find_knee_frequencies(
            y, TR, self.sequence_length, self.intermediate_vec,
            lz_p0=[900, 0.05],
            lz_bounds=([0, 0.01], [1200, 0.1]))

        ultralow, low, high = _filter_three_bands(
            y, TR, f1, f2, self.filtering_type)

        # Always pad to match pretraining length (464)
        high = F.pad(torch.from_numpy(high),
                     (pad // 2, pad // 2), 'constant', 0).T.float()
        low = F.pad(torch.from_numpy(low),
                    (pad // 2, pad // 2), 'constant', 0).T.float()
        ultralow = F.pad(torch.from_numpy(ultralow),
                         (pad // 2, pad // 2), 'constant', 0).T.float()

        return {
            'fmri_highfreq_sequence': high,
            'fmri_lowfreq_sequence': low,
            'fmri_ultralowfreq_sequence': ultralow,
            'subject': subj,
            'subject_name': subj_name,
            self.target: target,
        }


# ─── UKB ──────────────────────────────────────────────────────────────────────

class UKB_fMRI_timeseries(BaseDataset):
    """UK Biobank dataset. Used for pretraining (target='reconstruction')
    and downstream tasks (sex, depression, fluid intelligence)."""

    TR = 0.735  # seconds

    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('ukb_path')
        self.meta_data = pd.read_csv(
            os.path.join(kwargs.get('base_path'), 'data', 'metadata',
                         'UKB_phenotype_gps_fluidint.csv'))

        valid_sub = list(map(int, os.listdir(self.data_dir)))

        if self.target != 'reconstruction':
            non_na = self.meta_data[['eid', self.target]].dropna(axis=0)
            subjects = list(set(non_na['eid']) & set(valid_sub))
        else:
            subjects = valid_sub

        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
            self.mean = cont_mean
            self.std = cont_std

        for i, subject in enumerate(subjects):
            if self.fine_tune_task == 'regression':
                target = torch.tensor(
                    (self.meta_data.loc[self.meta_data['eid'] == subject,
                                        self.target].values[0]
                     - cont_mean) / cont_std).float()
            elif self.fine_tune_task == 'binary_classification':
                target = torch.tensor(
                    self.meta_data.loc[self.meta_data['eid'] == subject,
                                       self.target].values[0])
            else:  # reconstruction
                target = torch.tensor(0)

            if self.intermediate_vec == 360:
                j = str(subject) + '_20227_2_0'
                k = str(subject) + '_20227_3_0'
                path_j = os.path.join(self.data_dir, j, 'hcp_mmp1_360_' + j + '.npy')
                path_k = os.path.join(self.data_dir, k, 'hcp_mmp1_360_' + k + '.npy')
                path_to_fMRIs = path_j if os.path.exists(path_j) else path_k
            elif self.intermediate_vec == 400:
                path_to_fMRIs = os.path.join(
                    self.data_dir, str(subject),
                    'schaefer_400Parcels_17Networks_' + str(subject) + '.npy')

            self.index_l.append((i, subject, path_to_fMRIs, target))

    def __len__(self):
        return len(self.index_l)

    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]

        # Skip first 20 dummy scans
        y = np.load(path_to_fMRIs)[20:20 + self.sequence_length].T

        f1, f2 = _find_knee_frequencies(
            y, self.TR, self.sequence_length, self.intermediate_vec,
            lz_p0=[900, 0.05],
            lz_bounds=([0, 0.01], [1200, 0.1]))

        ultralow, low, high = _filter_three_bands(
            y, self.TR, f1, f2, self.filtering_type)

        high = torch.from_numpy(high).T.float()
        low = torch.from_numpy(low).T.float()
        ultralow = torch.from_numpy(ultralow).T.float()

        return {
            'fmri_highfreq_sequence': high,
            'fmri_lowfreq_sequence': low,
            'fmri_ultralowfreq_sequence': ultralow,
            'subject': subj,
            'subject_name': subj_name,
            self.target: target,
        }


# ─── ABCD ─────────────────────────────────────────────────────────────────────

class ABCD_fMRI_timeseries(BaseDataset):
    """Adolescent Brain Cognitive Development (ABCD) dataset.
    Targets: sex, ADHD_label, depression, fluid_intelligence, reconstruction."""

    TR = 0.8  # seconds

    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('abcd_path')

        if self.target == 'depression':
            self.meta_data = pd.read_csv(
                os.path.join(kwargs.get('base_path'), 'data', 'metadata',
                             'ABCD_5_1_KSADS_raw_MDD_ANX_CorP_pp_pres_ALL.csv'))
            self.meta_data['subjectkey'] = [
                i.split('-')[1] for i in self.meta_data['subjectkey']]
            self.target = 'MDD_pp'
        else:
            self.meta_data = pd.read_csv(
                os.path.join(kwargs.get('base_path'), 'data', 'metadata',
                             'ABCD_matched.csv'))

        valid_sub = [i.split('-')[1] for i in os.listdir(self.data_dir)]

        if self.target != 'reconstruction':
            non_na = self.meta_data[['subjectkey', self.target]].dropna(axis=0)
            subjects = list(set(non_na['subjectkey']) & set(valid_sub))
        else:
            subjects = valid_sub

        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
            self.mean = cont_mean
            self.std = cont_std

        for i, subject in enumerate(subjects):
            if self.intermediate_vec == 360:
                path_to_fMRIs = os.path.join(
                    self.data_dir, 'sub-' + subject,
                    'hcp_mmp1_sub-' + subject + '.npy')
            elif self.intermediate_vec == 400:
                path_to_fMRIs = os.path.join(
                    self.data_dir, 'sub-' + subject,
                    'schaefer_sub-' + subject + '.npy')

            if self.fine_tune_task == 'regression':
                target = torch.tensor(
                    (self.meta_data.loc[
                        self.meta_data['subjectkey'] == subject,
                        self.target].values[0] - cont_mean) / cont_std).float()
            elif self.fine_tune_task == 'binary_classification':
                target = torch.tensor(
                    self.meta_data.loc[
                        self.meta_data['subjectkey'] == subject,
                        self.target].values[0])
            else:  # reconstruction
                target = torch.tensor(0)

            self.index_l.append((i, subject, path_to_fMRIs, target))

    def __len__(self):
        return len(self.index_l)

    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]

        y = np.load(path_to_fMRIs)[:self.sequence_length].T  # [ROI, seq_len]

        if self.transfer_learning or self.finetune_test:
            pad = 464 - self.sequence_length

        f1, f2 = _find_knee_frequencies(
            y, self.TR, self.sequence_length, self.intermediate_vec,
            lz_p0=[1, 0.05],
            lz_bounds=([0, 0.01], [15, 0.1]))

        ultralow, low, high = _filter_three_bands(
            y, self.TR, f1, f2, self.filtering_type)

        if self.transfer_learning or self.finetune_test:
            high = F.pad(torch.from_numpy(high),
                         (pad // 2, pad // 2), 'constant', 0).T.float()
            low = F.pad(torch.from_numpy(low),
                        (pad // 2, pad // 2), 'constant', 0).T.float()
            ultralow = F.pad(torch.from_numpy(ultralow),
                             (pad // 2, pad // 2), 'constant', 0).T.float()
        else:
            high = torch.from_numpy(high).T.float()
            low = torch.from_numpy(low).T.float()
            ultralow = torch.from_numpy(ultralow).T.float()

        return {
            'fmri_highfreq_sequence': high,
            'fmri_lowfreq_sequence': low,
            'fmri_ultralowfreq_sequence': ultralow,
            'subject': subj,
            'subject_name': subj_name,
            self.target: target,
        }
