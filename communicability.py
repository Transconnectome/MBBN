import nitime
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer
from multiprocessing import Pool, cpu_count
from lmfit import Model
import scipy.stats as stats
import pywt
import networkx as nx
import args

def get_arguments(base_path = os.getcwd()):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROI_num', type=int,default=400, choices=[180, 400])
    parser.add_argument('--ukb_path', default='/scratch/connectome/stellasybae/UKB_ROI') ## labserver
    parser.add_argument('--abcd_path', default='/scratch/connectome/stellasybae/ABCD_ROI/7.ROI') ## labserver
    parser.add_argument('--dataset_name', type=str, choices=['ABCD', 'UKB'], default="UKB")
    parser.add_argument('--base_path', type=str, default=os.getcwd())
    args = parser.parse_args()
        
    return args

def wavelet_corr_mat(signal):
    # signal shape :  (ROI_num, seq_len)

    # wavelet transformation
    coeffs = pywt.dwt(signal, 'db1')  # 'db1' =  Daubechies wavelet
    cA, cD = coeffs  # cA: Approximation Coefficients, cD: etail Coefficients

    return np.corrcoef(cA)

def create_network(correlation_matrix, threshold=0.2):
    # Generate graph whose size is equivalent to correlation matrix
    G = nx.Graph()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i+1, correlation_matrix.shape[1]):
            # add edge when correlation coefficient > threshold.
            if np.abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)
    return G

if args.dataset_name == 'ABCD':
    data_dir = args.abcd_path
    TR = 0.8
    seq_len = 348
    subject = open(f'{args.base_path}/splits/ABCD/ABCD_reconstruction_ROI_{args.ROI_num}_seq_len_{seq_len}_split1.txt', 'r').readlines()
    subject = [x[:-1] for x in subject]
    subject.remove('train_subjects')
    subject.remove('val_subjects')
    subject.remove('test_subjects')


elif args.dataset_name == 'UKB':
    data_dir = args.ukb_path
    TR = 0.735
    seq_len = 464
    subject = open(f'{args.base_path}/splits/UKB/UKB_reconstruction_ROI_{args.ROI_num}_seq_len_{seq_len}_split1.txt', 'r').readlines()
    subject = [x[:-1] for x in subject]
    subject.remove('train_subjects')
    subject.remove('val_subjects')
    subject.remove('test_subjects')

if args.ROI_num == 400:
    ROI_name = 'Schaefer400'
elif args.ROI_num == 180:
    ROI_name = 'HCPMMP1'
    
subject = subject[:-1]
print('number of subject', len(subject))
num_processes = cpu_count()
print('number of processes', num_processes)

n = args.ROI_num
high_comm_mat_whole = np.zeros((n, n))
low_comm_mat_whole = np.zeros((n, n))
ultralow_comm_mat_whole = np.zeros((n, n))


def main(sub):
    try:
        path_to_fMRIs = os.path.join(data_dir, sub, 'schaefer_400Parcels_17Networks_'+sub+'.npy')
        y = np.load(path_to_fMRIs)[20:20+seq_len].T

        sample_whole = np.zeros((seq_len))
        for i in range(n):
            sample_whole+=y[i]

        sample_whole /= n    

        T = TimeSeries(sample_whole, sampling_interval=TR)
        S_original = SpectralAnalyzer(T)

        # Lorentzian function fitting
        xdata = np.array(S_original.spectrum_fourier[0][1:])
        ydata = np.abs(S_original.spectrum_fourier[1][1:])



        sample_whole = np.zeros(self.sequence_length,)
            for i in range(self.intermediate_vec):
                sample_whole+=y[i]

            sample_whole /= self.intermediate_vec    

            T = TimeSeries(sample_whole, sampling_interval=TR)
            S_original = SpectralAnalyzer(T)
            
            xdata = np.array(S_original.spectrum_fourier[0][1:])
            ydata = np.abs(S_original.spectrum_fourier[1][1:])

            
            # lmfit model setting
            model = Model(lorentzian_function)
            params = model.make_params()

            # Parameter initialization
            params['s0'].set(value=900, min=0.0, max=1200.0)
            params['f1'].set(value=0.05, min=0.01, max=0.1)

            # Fitting
            result = model.fit(ydata, params, x=xdata,
                               method='differential_evolution',
                               max_nfev=20000)

            f1 = result.params['f1'].value

            model = Model(spline_multifractal)
            params = model.make_params()

            params['beta_low'].set(value=-1.2, min=-5, max=-0.1)
            params['beta_high'].set(value=-0.5, min=-5, max=-0.1)
            params['A'].set(value=10, min=1, max=30) 
            params['f2'].set(value=0.08, min=f1+0.001, max=0.2)
            params['smoothness'].set(value=0.25, min=0.001, max=1.0) 

            # Fitting 실행
            result = model.fit(ydata[knee:], params, x=xdata[knee:],
                       method='differential_evolution',
                       max_nfev=20000)
            f2 = result.params['f2'].value
                

        # 01 high ~ (low+ultralow)
        T1 = TimeSeries(y, sampling_interval=TR)
        S_original1 = SpectralAnalyzer(T1)
        FA1 = FilterAnalyzer(T1, lb= f2)
        high = stats.zscore(FA1.fir.data, axis=1)
        ultralow_low = FA1.data-FA1.fir.data

        # 02 low ~ ultralow
        T2 = TimeSeries(ultralow_low, sampling_interval=TR)
        S_original2 = SpectralAnalyzer(T2)
        FA2 = FilterAnalyzer(T2, lb=f1)
        low = stats.zscore(FA2.fir.data, axis=1)
        ultralow = stats.zscore(FA2.data-FA2.fir.data, axis=1)

        high_G = create_network(wavelet_corr_mat(high))
        high_comm = nx.communicability(high_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = high_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        high_comm_mat_whole=communicability_matrix

        low_G = create_network(wavelet_corr_mat(low))
        low_comm = nx.communicability(low_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = low_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        low_comm_mat_whole=communicability_matrix

        ultralow_G = create_network(wavelet_corr_mat(ultralow))
        ultralow_comm = nx.communicability(ultralow_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = ultralow_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        ultralow_comm_mat_whole=communicability_matrix
    except:
        high_comm_mat_whole=np.zeros((n, n))
        low_comm_mat_whole=np.zeros((n, n))
        ultralow_comm_mat_whole=np.zeros((n, n))
    
    return high_comm_mat_whole, low_comm_mat_whole, ultralow_comm_mat_whole


pool = Pool(num_processes)
results = pool.map(main, subject)

sub_num = len(subject)

high_comm_mat_whole = sum([results[i][0] for i in range(sub_num)]) / len(subject)
low_comm_mat_whole = sum([results[i][1] for i in range(sub_num)]) / len(subject)
ultralow_comm_mat_whole = sum([results[i][2] for i in range(sub_num)]) / len(subject)


np.save(f'./data/comnunicability/{args.dataset_name}_new_high_comm_ROI_order_{ROI_name}.npy', np.argsort(np.sum(high_comm_mat_whole, axis=1)))
np.save(f'./data/comnunicability/{args.dataset_name}_new_low_comm_ROI_order_{ROI_name}.npy', np.argsort(np.sum(low_comm_mat_whole, axis=1)))
np.save(f'./data/comnunicability/{args.dataset_name}_new_ultralow_comm_ROI_order_{ROI_name}.npy', np.argsort(np.sum(ultralow_comm_mat_whole, axis=1)))
# last ROI has highest communicability