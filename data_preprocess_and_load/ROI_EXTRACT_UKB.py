#######The final version (with for loop)##########
import os
import nilearn
from nilearn import input_data 
import nibabel as nib
import numpy as np
import argparse
import time
from multiprocessing import Pool


parser = argparse.ArgumentParser(description = "arguments")
parser.add_argument('--subject_list', default = './rsfMRI_sub_list_0.txt')
args = parser.parse_args()


file_atlas = '/global/cfs/cdirs/m4244/stella/divfreqBERT/data/atlas/HCP-MMP1_on_MNI152_adult.nii.gz'

sub_list = np.loadtxt(args.subject_list, delimiter=',',dtype='str')

input_dir = '/global/cfs/cdirs/m4244/registration/20227_1_MNI/'
save_dir = '/global/cfs/cdirs/m4244/stella/SWIFT_baseline/UKB/'

def extract_atlas_timeseries(file, img_input_cleaned, file_atlas):
    TR = 0.735
    masker = input_data.NiftiLabelsMasker(labels_img = file_atlas, detrend=True, low_pass = 0.08, t_r = TR)
    masker.fit()

    roi= masker.transform(img_input_cleaned)
    return roi

def main(sub):
    #sub = file.split('_func')[0]
    folder = save_dir + sub
    os.makedirs(folder, exist_ok = True)
    # raw image
    img_input_cleaned= nib.load(input_dir+sub+'_20227_2_0_rsfMRI_MNI_space.nii.gz')
    data = extract_atlas_timeseries(sub, img_input_cleaned, file_atlas)
    print('shape of', sub, 'is', data.shape)
    np.save(folder+'/hcp_mmp1_'+sub+'.npy', data)
    
pool = Pool(4)
output = pool.map(main, sub_list)