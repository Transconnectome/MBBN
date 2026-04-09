#######The final version (with for loop)##########
import os
import nilearn
from nilearn import input_data 
import nibabel as nib
import numpy as np
import argparse
import time
from multiprocessing import Pool
import pickle


parser = argparse.ArgumentParser(description = "arguments")
parser.add_argument('--subject_list', default = './rsfMRI_sub_list_0.txt')
args = parser.parse_args()


file_atlas = '/scratch/connectome/stellasybae/for_share/MBBN/data/atlas/HCPMMP1_asymmetric_for_ABIDE.nii.gz'

with open(args.subject_list, "r") as f:
    sub_list = [line.strip() for line in f.readlines()]  # strip trailing newlines

input_dir = '/scratch/connectome/stellasybae/ABIDE_4D'
save_dir = '/scratch/connectome/stellasybae/ABIDE_ROI/'

def extract_atlas_timeseries(sub_id, fullname, img_input_cleaned, file_atlas):
    print('fullname is', fullname)
    # sub_id format: e.g. 0051459
    if 'CALTECH' in fullname:
        TR = 2
    elif 'CMU' in fullname:
        TR = 2
    elif 'KKI' in fullname:
        TR = 2.5
    elif 'LEUVEN' in fullname:
        TR = 1.66
    elif 'MAX_MUN' in fullname:
        TR = 3
    elif 'NYU' in fullname:
        TR = 2
    elif 'OHSU' in fullname:
        TR = 2.5
    elif 'OLIN' in fullname:
        TR = 1.5
    elif 'PITT' in fullname:
        TR = 1.5
    elif 'SBL' in fullname:
        TR = 2.2
    elif 'SDSU' in fullname:
        TR = 2
    elif 'STANFORD' in fullname:
        TR = 2 
    elif 'TRINITY' in fullname:
        TR = 2 
    elif 'UCLA' in fullname:
        TR = 3 
    elif 'UM' in fullname:
        TR = 2 
    elif 'USM' in fullname:
        TR = 2
    elif 'YALE' in fullname:
        TR = 2
    else:
        TR = 2
    masker = input_data.NiftiLabelsMasker(labels_img = file_atlas, detrend=True, low_pass = 0.08, t_r = TR)
    masker.fit()

    roi= masker.transform(img_input_cleaned)
    return roi

def main(filename):
    fullname = filename.split('_func')[0]
    sub = filename.split('_func')[0].split('_')[-1]
    folder = save_dir + sub
    os.makedirs(folder, exist_ok = True)
    # raw image
    img_input_cleaned= nib.load(os.path.join(input_dir, filename))
    data = extract_atlas_timeseries(sub, fullname, img_input_cleaned, file_atlas)
    print('shape of', sub, 'is', data.shape)
    np.save(folder+'/hcp_mmp1_360_'+sub+'.npy', data)
    
pool = Pool(4)
output = pool.map(main, sub_list)