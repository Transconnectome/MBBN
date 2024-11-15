import numpy as np
import pandas as pd
import scipy.io
import random
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
import nibabel as nib
import scipy

from scipy import stats

import torch.nn.functional as F
import nitime
from scipy.optimize import curve_fit

import pickle

import warnings
warnings.filterwarnings("ignore")

# Import the time-series objects:
from nitime.timeseries import TimeSeries

# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer

def lorentzian_function(x, s0, corner):
    return (s0*corner**2) / (x**2 + corner**2)

def multi_fractal_function(x, beta_low, beta_high, A, B, corner):
    return np.where(x < corner, A * x**beta_low, B * x**beta_high)

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def register_args(self,**kwargs):
        self.index_l = []
        self.target = kwargs.get('target')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.dataset_name = kwargs.get('dataset_name')
        self.fmri_type = kwargs.get('fmri_type')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.seq_len = kwargs.get('sequence_length')
        self.lorentzian = kwargs.get('divide_by_lorentzian')
        self.fmri_dividing_type = kwargs.get('fmri_dividing_type')
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.filtering_type = kwargs.get('filtering_type')
        self.sequence_length = kwargs.get('sequence_length')
        self.use_raw_knee = kwargs.get('use_raw_knee')
        self.seq_part = kwargs.get('seq_part')
        self.use_high_freq = kwargs.get('use_high_freq')
        self.pretrained_model_weights_path = kwargs.get('pretrained_model_weights_path')
        self.finetune = kwargs.get('finetune')
        self.transfer_learning =  bool(self.pretrained_model_weights_path) or self.finetune
        self.finetune_test = kwargs.get('finetune_test') # test phase of finetuning task
        
        

class ABIDE_fMRI_timeseries(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('abide_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABIDE1+2_meta.csv'))
        self.site_meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABIDE1_pheno_and_sites.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        self.sex = kwargs.get('sex')
        
        # removing samples whose target value is NaN.
        
        if self.target == 'sex':
            non_na = self.meta_data[['SUB_ID', 'SEX']].dropna(axis=0)
            subjects = list(non_na['SUB_ID']) # form : Pitt_0050004
            subjects = list(map(str, subjects)) 
        elif self.target == 'ASD':
            if self.sex == 'both':
                non_na = self.meta_data[['SUB_ID', 'DX_GROUP']].dropna(axis=0)
            elif self.sex == 'male':
                df = self.meta_data[['SUB_ID', 'SEX', 'DX_GROUP']].dropna(axis=0)
                non_na = df[df['SEX'] == 1]
            elif self.sex == 'female':
                df = self.meta_data[['SUB_ID', 'SEX', 'DX_GROUP']].dropna(axis=0)
                non_na = df[df['SEX'] == 2]    
                
        else:
            non_na = self.meta_data[['SUB_ID', self.target]].dropna(axis=0)
            
        data_list = []

        for sub in os.listdir(self.data_dir):
            data_list.append(os.path.join(self.data_dir, sub, 'schaefer_400Parcels_17Networks_'+sub+'.npy'))
        
        unified_name_list = [i[2:] if i.startswith('00') else i for i in os.listdir(self.data_dir)]
        # if starts with 005~, then ABIDE1. else, ABIDE2.
        
        non_na = self.meta_data.dropna(axis=0)
        valid_sub = set([str(i) for i in non_na['SUB_ID']]) & set(unified_name_list) # now starts with 5, 2
        
        for i, filename in enumerate(data_list):
            sub = filename.split('/')[-2]
            if sub.startswith('00'):
                # ABIDE 1
                subid = sub[2:] #sub is 0051316, subid is 51316
                site = list(self.site_meta_data[self.site_meta_data['SUB_ID'] == int(subid)]['SITE_ID'])[0]
            else:
                subid = sub
                site = 'ABIDE2'
            if subid in valid_sub:
                if self.target == 'sex':
                    target = non_na.loc[non_na['SUB_ID']==int(subid), 'SEX'].values[0]
                elif self.target == 'ASD':
                    target = non_na.loc[non_na['SUB_ID']==int(subid), 'DX_GROUP'].values[0]
                target = 1.0 if target == 2 else 0.0
                target = torch.tensor(target)
                
                self.index_l.append((i, sub, filename, target, site))
                
    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target, site = self.index_l[index]

        if self.seq_part=='tail':
            y = np.load(path_to_fMRIs)[-self.sequence_length:].T # [ROI, seq_len]
        elif self.seq_part=='head':
            y = np.load(path_to_fMRIs)[20:20+self.sequence_length].T # [ROI, seq_len]
        
        ts_length = y.shape[1]
        pad = self.sequence_length-ts_length
        
        if self.transfer_learning or self.finetune_test:
            # standard length : 464 (UKB) - because I pretrained divfreqBERT with UKB!
            pad = 464 - self.sequence_length

        if 'CALTECH' in site:
            TR = 2
        elif 'CMU' in site:
            TR = 2
        elif 'KKI' in site:
            TR = 2.5
        elif 'LEUVEN' in site:
            TR = 1.66
        elif 'MAX_MUN' in site:
            TR = 3
        elif 'NYU' in site:
            TR = 2
        elif 'OHSU' in site:
            TR = 2.5
        elif 'OLIN' in site:
            TR = 1.5
        elif 'PITT' in site:
            TR = 1.5
        elif 'SBL' in site:
            TR = 2.2
        elif 'SDSU' in site:
            TR = 2
        elif 'STANFORD' in site:
            TR = 2 
        elif 'TRINITY' in site:
            TR = 2 
        elif 'UCLA' in site:
            TR = 3 
        elif 'UM' in site:
            TR = 2 
        elif 'USM' in site:
            TR = 2
        elif 'YALE' in site:
            TR = 2
        else:
            TR = 3 # ABIDE 2


        if self.lorentzian:
        
            '''
            get knee frequency
            '''

            sample_whole = np.zeros(self.sequence_length,) # originally self.sequence_length.
            for i in range(self.intermediate_vec):
                sample_whole+=y[i]

            sample_whole /= self.intermediate_vec    

            T = TimeSeries(sample_whole, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)

            # Lorentzian function fitting (dividing ultralow ~ low)
            xdata = np.array(S_original.spectrum_fourier[0][1:])
            ydata = np.abs(S_original.spectrum_fourier[1][1:])

            # initial parameter setting
            p0 = [0, 0.006]
            param_bounds = ([-np.inf, 0], [np.inf, 1])

            # fitting Lorentzian function
            popt, pcov = curve_fit(lorentzian_function, xdata, ydata, p0=p0, maxfev = 5000, bounds=param_bounds)
            
            f1 = popt[1]
            
            knee = round(popt[1]/(1/(sample_whole.shape[0]*TR)))
            
            if knee <= 0:
                knee = 1
            
            # divide low ~ high
            if self.fmri_dividing_type == 'three_channels':
                # initial parameter setting
                p1 = [2, 1, 23, 25, 0.16]
                
                # fitting multifractal function
                popt_mo, pcov = curve_fit(multi_fractal_function, xdata[knee:], ydata[knee:], p0=p1, maxfev = 50000)
                pink = round(popt_mo[-1]/(1/(sample_whole.shape[0]*TR)))
                f2 = popt_mo[-1]

        # don't use Lorentzian function to divide frequencies
        else:
            if self.fmri_type == 'timeseries':
                pass
            else:
                ## don't use raw knee frequency!
                sample_whole = np.zeros(self.sequence_length,)
                for i in range(self.intermediate_vec):
                    sample_whole+=y[i]

                sample_whole /= self.intermediate_vec    

                T = TimeSeries(sample_whole, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)

                # random frequencies
                xdata = np.array(S_original.spectrum_fourier[0][1:])
                frequency_range = list(range(xdata.shape[0]))
                import random
                if self.fmri_dividing_type == 'three_channels':
                    a,b = random.sample(frequency_range, 2)
                    knee = min(a,b)
                    if knee == 0:
                        knee = 1
                    pink = max(a,b)
                    if pink == len(frequency_range)-1:
                        pink = len(frequency_range)-2
                elif self.fmri_dividing_type == 'two_channels':
                    knee = random.sample(frequency_range, 1)[0]
                    if knee == 0:
                        knee = 1
        
        if self.fmri_type == 'timeseries':
            y = scipy.stats.zscore(y, axis=1)
            y = torch.from_numpy(y).T.float()
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            y = scipy.stats.zscore(np.abs(S_original.spectrum_fourier[1]), axis=None) 
            y = torch.from_numpy(y).T.float()
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_low':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA2.filtered_boxcar.data, axis=1)
                
                low = torch.from_numpy(low).T.float()
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1)

                low = torch.from_numpy(low).T.float()
                
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_ultralow':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                ultralow = torch.from_numpy(ultralow).T.float()
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1)

                ultralow = torch.from_numpy(ultralow).T.float()
            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'time_domain_high':
            T1 = TimeSeries(y, sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
            high = scipy.stats.zscore(FA1.filtered_boxcar.data, axis=1)
            high = torch.from_numpy(high).T.float()
            ans_dict = {'fmri_sequence':high,'subject':subj,'subject_name':subj_name, self.target:target}
            
        elif self.fmri_type == 'divided_timeseries':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                if self.use_raw_knee:
                    FA1 = FilterAnalyzer(T1, lb = f2)
                else:
                    FA1 = FilterAnalyzer(T1, lb = S_original1.spectrum_fourier[0][pink])
                high = stats.zscore(FA1.filtered_boxcar.data, axis=1)
                ultralow_low = FA1.data-FA1.filtered_boxcar.data
                    
                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=f1)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                if self.filtering_type == 'FIR':
                    low = stats.zscore(FA2.fir.data, axis=1)
                    ultralow = stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = stats.zscore(FA2.filtered_boxcar.data, axis=1)
                    ultralow = stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                    
                # DO PADDING ALWAYS
                high = F.pad(torch.from_numpy(high), (pad//2, pad//2), "constant", 0).T.float()
                low = F.pad(torch.from_numpy(low), (pad//2, pad//2), "constant", 0).T.float()
                ultralow = F.pad(torch.from_numpy(ultralow), (pad//2, pad//2), "constant", 0).T.float()
                    
#                 if self.transfer_learning or self.finetune_test:
#                     # do padding! high : (ROI, time length)
#                     high = F.pad(torch.from_numpy(high), (pad//2, pad//2), "constant", 0).T.float()
#                     low = F.pad(torch.from_numpy(low), (pad//2, pad//2), "constant", 0).T.float()
#                     ultralow = F.pad(torch.from_numpy(ultralow), (pad//2, pad//2), "constant", 0).T.float()

                    
#                 else: 
#                     high = torch.from_numpy(high).T.float()
#                     low = torch.from_numpy(low).T.float()
#                     ultralow = torch.from_numpy(ultralow).T.float()
                
                if self.use_high_freq:
                    ans_dict= {'fmri_highfreq_sequence':high, 'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
                else:
                    ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
            
            else: # two channels
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=f1)
                else:
                    print(knee)
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1)
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1)
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1)

                low = torch.from_numpy(low).T.float()
                ultralow = torch.from_numpy(ultralow).T.float()

                ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}

       
            
        elif self.fmri_type == 'frequency_domain_low':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])
            T1 = TimeSeries((FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            
            # complex number -> real number (amplitude)
            low = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            pad_l = self.sequence_length//2 - low.shape[1]
            
            low = torch.from_numpy(low).T.float()
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency_domain_ultralow':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

            T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)

            ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            ultralow = torch.from_numpy(ultralow).T.float() 

            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'frequency_domain_high':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][pink])
            T1 = TimeSeries((FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            
            # complex number -> real number (amplitude)
            high = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            pad_h = self.sequence_length//2 - high.shape[1]
            
            high = torch.from_numpy(high).T.float()
            ans_dict = {'fmri_sequence':high,'subject':subj,'subject_name':subj_name, self.target:target}
        
        
        return ans_dict
    
    

class UKB_fMRI_timeseries(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('ukb_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','UKB_phenotype_gps_fluidint.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
                
        valid_sub = os.listdir(kwargs.get('ukb_path'))
        valid_sub = list(map(int, valid_sub))
        
        if self.target != 'reconstruction':
            non_na = self.meta_data[['eid',self.target]].dropna(axis=0)
            subjects = set(non_na['eid']) & set(valid_sub)
        else:
            subjects = valid_sub
                
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
            self.mean = cont_mean
            self.std = cont_std
            
        for i, subject in enumerate(subjects):
            # Normalization
            if self.fine_tune_task == 'regression':
                target = torch.tensor((self.meta_data.loc[self.meta_data['eid']==subject,self.target].values[0] - cont_mean) / cont_std)
                target = target.float()
            elif self.fine_tune_task == 'binary_classification':
                target = torch.tensor(self.meta_data.loc[self.meta_data['eid']==subject,self.target].values[0]) 
            else:
                if self.target == 'reconstruction': # for transformer reconstruction
                    target = torch.tensor(0)
                    
            if self.intermediate_vec == 180:
                path_to_fMRIs = os.path.join(self.data_dir, str(subject), 'hcp_mmp1_'+str(subject)+'.npy')
            elif self.intermediate_vec == 400:
                path_to_fMRIs = os.path.join(self.data_dir, str(subject), 'schaefer_400Parcels_17Networks_'+str(subject)+'.npy')
                
                
            self.index_l.append((i, subject, path_to_fMRIs, target))           

            

    def __len__(self):
        N = len(self.index_l)
        return N
    
        
    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]
        
        if self.seq_part=='tail':
            y = np.load(path_to_fMRIs)[-self.sequence_length:].T # [180, seq_len]
        elif self.seq_part=='head':
            y = np.load(path_to_fMRIs)[20:20+self.sequence_length].T # [180, seq_len]
        
        ts_length = y.shape[1]
        pad = self.sequence_length - ts_length # 어차피 0일거임..

        TR = 0.735
        if self.lorentzian:
        
            '''
            get knee frequency
            '''

            sample_whole = np.zeros(self.sequence_length,)
            for i in range(self.intermediate_vec):
                sample_whole+=y[i]

            sample_whole /= self.intermediate_vec    

            T = TimeSeries(sample_whole, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)

            # Lorentzian function fitting
            xdata = np.array(S_original.spectrum_fourier[0][1:])
            ydata = np.abs(S_original.spectrum_fourier[1][1:])

            # initial parameter
            p0 = [0, 0.006]

            # fitting Lorentzian function
            popt, pcov = curve_fit(lorentzian_function, xdata, ydata, p0=p0, maxfev = 5000)
            
            f1 = popt[1]
            
            knee = round(popt[1]/(1/(sample_whole.shape[0]*TR)))
            
            if knee <= 0:
                knee = 1
            
            
            if self.fmri_dividing_type == 'three_channels':
                # initial parameter
                p1 = [2, 1, 23, 25, 0.16]
                # fitting multifractal function
                popt_mo, pcov = curve_fit(multi_fractal_function, xdata[knee:], ydata[knee:], p0=p1, maxfev = 50000)
                pink = round(popt_mo[-1]/(1/(sample_whole.shape[0]*TR)))
                f2 = popt_mo[-1]

        else:
            if self.fmri_type == 'timeseries':
                pass
            else:
                ## don't use raw knee frequency
                sample_whole = np.zeros(self.sequence_length,)
                for i in range(self.intermediate_vec):
                    sample_whole+=y[i]

                sample_whole /= self.intermediate_vec    

                T = TimeSeries(sample_whole, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)

                # Lorentzian function fitting
                xdata = np.array(S_original.spectrum_fourier[0][1:])
                frequency_range = list(range(xdata.shape[0]))
                import random
                if self.fmri_dividing_type == 'three_channels':
                    a,b = random.sample(frequency_range, 2)
                    knee = min(a,b)
                    if knee == 0:
                        knee = 1
                    pink = max(a,b)
                    if pink == len(frequency_range)-1:
                        pink = len(frequency_range)-2
                elif self.fmri_dividing_type == 'two_channels':
                    knee = random.sample(frequency_range, 1)[0]
                    if knee == 0:
                        knee = 1
        
        if self.fmri_type == 'timeseries':
            y = scipy.stats.zscore(y, axis=1)
            y = torch.from_numpy(y).T.float()
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            y = scipy.stats.zscore(np.abs(S_original.spectrum_fourier[1]), axis=None)
            y = torch.from_numpy(y).T.float()
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_low':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA2.filtered_boxcar.data, axis=1)
                
                low = torch.from_numpy(low).T.float()
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1) 

                low = torch.from_numpy(low).T.float()
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_ultralow':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow) 
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                ultralow = torch.from_numpy(ultralow).T.float()
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1)

                ultralow = torch.from_numpy(ultralow).T.float()
            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'time_domain_high':
            T1 = TimeSeries(y, sampling_interval=0.8)
            S_original1 = SpectralAnalyzer(T1)
            FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
            high = scipy.stats.zscore(FA1.filtered_boxcar.data, axis=1)
            high = torch.from_numpy(high).T.float()
            ans_dict = {'fmri_sequence':high,'subject':subj,'subject_name':subj_name, self.target:target}
            
        elif self.fmri_type == 'divided_timeseries':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                if self.use_raw_knee:
                    FA1 = FilterAnalyzer(T1, lb = f2)
                else:
                    FA1 = FilterAnalyzer(T1, lb = S_original1.spectrum_fourier[0][pink])
                high = stats.zscore(FA1.filtered_boxcar.data, axis=1)
                ultralow_low = FA1.data-FA1.filtered_boxcar.data
                    
                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=f1)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                if self.filtering_type == 'FIR':
                    low = stats.zscore(FA2.fir.data, axis=1)
                    ultralow = stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = stats.zscore(FA2.filtered_boxcar.data, axis=1)
                    ultralow = stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                    
                high = torch.from_numpy(high).T.float()
                low = torch.from_numpy(low).T.float()
                ultralow = torch.from_numpy(ultralow).T.float()
                
                if self.use_high_freq:
                    ans_dict= {'fmri_highfreq_sequence':high, 'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
                else:
                    ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
            
            else: # two channels    
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=f1)
                else:
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1)
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1)
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1)

                low = torch.from_numpy(low).T.float()
                ultralow = torch.from_numpy(ultralow).T.float()

                ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}

                
        elif self.fmri_type == 'frequency_domain_low':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])
            T1 = TimeSeries((FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            
            # complex number -> real number (amplitude)
            low = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            pad_l = self.sequence_length//2 - low.shape[1]
            
            low = torch.from_numpy(low).T.float()
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency_domain_ultralow':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

            T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)

            ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            ultralow = torch.from_numpy(ultralow).T.float()   

            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
       
        elif self.fmri_type == 'frequency_domain_high':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][pink])
            T1 = TimeSeries((FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            
            # complex number -> real number (amplitude)
            high = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            pad_h = self.sequence_length//2 - high.shape[1]
            
            high = torch.from_numpy(high).T.float()
            ans_dict = {'fmri_sequence':high,'subject':subj,'subject_name':subj_name, self.target:target}

        return ans_dict
    

    
class ABCD_fMRI_timeseries(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('abcd_path')
        if self.target == 'depression':
            self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_5_1_KSADS_raw_MDD_ANX_CorP_pp_pres_ALL.csv'))
            self.meta_data['subjectkey'] = [i.split('-')[1] for i in self.meta_data['subjectkey']]
            self.target = 'MDD_pp'
        else:
            self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        
        valid_sub = [i.split('-')[1] for i in os.listdir(self.data_dir)]
        
        if self.target != 'reconstruction':
            non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
            subjects = list(non_na['subjectkey'])
            subjects = list(set(subjects) & set(valid_sub))
        else:
            subjects = valid_sub
                               
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
            self.mean = cont_mean
            self.std = cont_std
        for i, subject in enumerate(subjects):
            
            if self.intermediate_vec == 180:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'hcp_mmp1_180_sub-'+subject+'.npy')
            elif self.intermediate_vec == 360:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'hcp_mmp1_sub-'+subject+'.npy')
            elif self.intermediate_vec == 400:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'schaefer_sub-'+subject+'.npy') 
            elif self.intermediate_vec == 246:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'Brainnectome_246_sub-'+subject+'.npy') 
            elif self.intermediate_vec == 200:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'craddock200_sub-'+subject+'.npy')
            elif self.intermediate_vec == 333:
                path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'gordon_sub-'+subject+'.npy')            
            
            # Normalization
            if self.fine_tune_task == 'regression':
                target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                target = target.float()
            elif self.fine_tune_task == 'binary_classification':
                target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0])
            else:
                if self.target == 'reconstruction': # for transformer reconstruction
                    target = torch.tensor(0)
                    
            self.index_l.append((i, subject, path_to_fMRIs, target))

            

    def __len__(self):
        N = len(self.index_l)
        return N
    
        
    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]
        if self.seq_part=='tail':
            y = np.load(path_to_fMRIs)[-self.sequence_length:].T # [180, seq_len]
        elif self.seq_part=='head':
            y = np.load(path_to_fMRIs)[:self.sequence_length].T # [180, seq_len]
        
        if self.transfer_learning or self.finetune_test:
            # pad to 464 (I pretrained my model with UKB)
            pad = 464 - self.sequence_length

        TR = 0.8
        if self.lorentzian:
        
            '''
            get knee frequency
            '''

            sample_whole = np.zeros(self.sequence_length,)
            for i in range(self.intermediate_vec):
                sample_whole+=y[i]

            sample_whole /= self.intermediate_vec    

            T = TimeSeries(sample_whole, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            
            xdata = np.array(S_original.spectrum_fourier[0][1:])
            ydata = np.abs(S_original.spectrum_fourier[1][1:])

            # initialize parameters
            p0 = [0, 0.006]

            # Lorentzian function fitting
            popt, pcov = curve_fit(lorentzian_function, xdata, ydata, p0=p0, maxfev = 5000)
            
            f1 = popt[1]
            
            knee = round(popt[1]/(1/(sample_whole.shape[0]*TR)))
            
            if knee <= 0:
                knee = 1
            
            
            if self.fmri_dividing_type == 'three_channels':
                # initialize parameters
                p1 = [2, 1, 23, 25, 0.16]
                
                # multi-fractal function fitting
                popt_mo, pcov = curve_fit(multi_fractal_function, xdata[knee:], ydata[knee:], p0=p1, maxfev = 50000)
                pink = round(popt_mo[-1]/(1/(sample_whole.shape[0]*TR)))
                f2 = popt_mo[-1]



        else:
            if self.fmri_type == 'timeseries':
                pass
            else:
                ## don't use raw knee frequency
                sample_whole = np.zeros(self.sequence_length,)
                for i in range(self.intermediate_vec):
                    sample_whole+=y[i]

                sample_whole /= self.intermediate_vec    

                T = TimeSeries(sample_whole, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)

                xdata = np.array(S_original.spectrum_fourier[0][1:])
                frequency_range = list(range(xdata.shape[0]))
                import random
                if self.fmri_dividing_type == 'three_channels':
                    a,b = random.sample(frequency_range, 2)
                    knee = min(a,b)
                    if knee == 0:
                        knee = 1
                    pink = max(a,b)
                    if pink == len(frequency_range)-1:
                        pink = len(frequency_range)-2
                elif self.fmri_dividing_type == 'two_channels':
                    knee = random.sample(frequency_range, 1)[0]
                    if knee == 0:
                        knee = 1
                ##knee = self.sequence_length//self.knee_divisor
        
        if self.fmri_type == 'timeseries':
            y = scipy.stats.zscore(y, axis=1)
            y = torch.from_numpy(y).T.float()
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency':
            T = TimeSeries(y, sampling_interval=0.8)
            S_original = SpectralAnalyzer(T)
            y = scipy.stats.zscore(np.abs(S_original.spectrum_fourier[1]), axis=None)
            y = torch.from_numpy(y).T.float() # (184, 84)
            ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_low':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA2.filtered_boxcar.data, axis=1)
                
                low = torch.from_numpy(low).T.float()
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1)

                low = torch.from_numpy(low).T.float()
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'time_domain_ultralow':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
                ultralow_low = FA1.data-FA1.filtered_boxcar.data

                # 02 low ~ ultralow
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=raw_knee)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                    
                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                ultralow = torch.from_numpy(ultralow).T.float()
            
            else:
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=raw_knee)
                else:    
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1)

                ultralow = torch.from_numpy(ultralow).T.float()
            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        elif self.fmri_type == 'time_domain_high':
            T1 = TimeSeries(y, sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            FA1 = FilterAnalyzer(T1, lb= S_original1.spectrum_fourier[0][pink])
            high = scipy.stats.zscore(FA1.filtered_boxcar.data, axis=1)
            high = torch.from_numpy(high).T.float()
            ans_dict = {'fmri_sequence':high,'subject':subj,'subject_name':subj_name, self.target:target}
            
        elif self.fmri_type == 'divided_timeseries':
            if self.fmri_dividing_type == 'three_channels':
                # 01 high ~ (low+ultralow)
                T1 = TimeSeries(y, sampling_interval=TR)
                S_original1 = SpectralAnalyzer(T1)
                if self.use_raw_knee:
                    FA1 = FilterAnalyzer(T1, lb = f2)
                else:
                    FA1 = FilterAnalyzer(T1, lb = S_original1.spectrum_fourier[0][pink])
                high = stats.zscore(FA1.filtered_boxcar.data, axis=1)
                ultralow_low = FA1.data-FA1.filtered_boxcar.data
                    
                # 02 low ~ ultralow 뜯어내기
                T2 = TimeSeries(ultralow_low, sampling_interval=TR)
                S_original2 = SpectralAnalyzer(T2)
                if self.use_raw_knee:
                    FA2 = FilterAnalyzer(T2, lb=f1)
                else:    
                    FA2 = FilterAnalyzer(T2, lb= S_original2.spectrum_fourier[0][knee])
                if self.filtering_type == 'FIR':
                    low = stats.zscore(FA2.fir.data, axis=1)
                    ultralow = stats.zscore(FA2.data-FA2.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = stats.zscore(FA2.filtered_boxcar.data, axis=1)
                    ultralow = stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)
                
                if self.transfer_learning or self.finetune_test:
                    high = F.pad(torch.from_numpy(high), (pad//2, pad//2), "constant", 0).T.float()
                    low = F.pad(torch.from_numpy(low), (pad//2, pad//2), "constant", 0).T.float()
                    ultralow = F.pad(torch.from_numpy(ultralow), (pad//2, pad//2), "constant", 0).T.float()
                    
                else: 
                    high = torch.from_numpy(high).T.float()
                    low = torch.from_numpy(low).T.float()
                    ultralow = torch.from_numpy(ultralow).T.float()
                
                if self.use_high_freq:
                    ans_dict= {'fmri_highfreq_sequence':high, 'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
                else:
                    ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
            
            else: # two channels    
                T = TimeSeries(y, sampling_interval=TR)
                S_original = SpectralAnalyzer(T)
                if self.use_raw_knee:
                    FA = FilterAnalyzer(T, lb=f1)
                else:
                    FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1)
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1)
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1)

                low = torch.from_numpy(low).T.float() # (324, 180)
                ultralow = torch.from_numpy(ultralow).T.float()

                ans_dict= {'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}

                
        elif self.fmri_type == 'frequency_domain_low':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])
            T1 = TimeSeries((FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)
            
            # complex number -> real number (amplitude)
            low = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            pad_l = self.sequence_length//2 - low.shape[1]
            
            low = torch.from_numpy(low).T.float()
            ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}

        elif self.fmri_type == 'frequency_domain_ultralow':
            T = TimeSeries(y, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            FA = FilterAnalyzer(T, lb= S_original.spectrum_fourier[0][knee])

            T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=TR)
            S_original1 = SpectralAnalyzer(T1)

            ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T)
            ultralow = torch.from_numpy(ultralow).T.float()  

            ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}


        return ans_dict