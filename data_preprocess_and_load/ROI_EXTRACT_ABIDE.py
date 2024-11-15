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


file_atlas = '../data/atlas/HCP-MMP1_on_MNI152_Pediatric.nii.gz'

#sub_list = np.loadtxt(args.subject_list, delimiter=',',dtype='str') # Caltech_0051459 형식
with open(args.subject_list, 'rb') as f:
    sub_dict = pickle.load(f)

sub_list = list(sub_dict.keys())

input_dir = '/storage/bigdata/ABIDE/fmri/ABIDE1/'
save_dir = '/storage/bigdata/ABIDE/fmri/ROI_DATA/'

def extract_atlas_timeseries(sub_id, sub_dict, img_input_cleaned, file_atlas):
    # sub_id : 0051459 형식
    if 'CALTECH' in sub_dict[sub_id]:
        TR = 2
    elif 'CMU' in sub_dict[sub_id]:
        TR = 2
    elif 'KKI' in sub_dict[sub_id]:
        TR = 2.5
    elif 'LEUVEN' in sub_dict[sub_id]:
        TR = 1.66
    elif 'MAX_MUN' in sub_dict[sub_id]:
        TR = 3
    elif 'NYU' in sub_dict[sub_id]:
        TR = 2
    elif 'OHSU' in sub_dict[sub_id]:
        TR = 2.5
    elif 'OLIN' in sub_dict[sub_id]:
        TR = 1.5
    elif 'PITT' in sub_dict[sub_id]:
        TR = 1.5
    elif 'SBL' in sub_dict[sub_id]:
        TR = 2.2
    elif 'SDSU' in sub_dict[sub_id]:
        TR = 2
    elif 'STANFORD' in sub_dict[sub_id]:
        TR = 2 
    elif 'TRINITY' in sub_dict[sub_id]:
        TR = 2 
    elif 'UCLA' in sub_dict[sub_id]:
        TR = 3 
    elif 'UM' in sub_dict[sub_id]:
        TR = 2 
    elif 'USM' in sub_dict[sub_id]:
        TR = 2
    elif 'YALE' in sub_dict[sub_id]:
        TR = 2 
    masker = input_data.NiftiLabelsMasker(labels_img = file_atlas, detrend=True, low_pass = 0.08, t_r = TR)
    masker.fit()

    roi= masker.transform(img_input_cleaned)
    return roi

def main(sub):
    folder = save_dir + sub
    os.makedirs(folder, exist_ok = True)
    # raw image
    img_input_cleaned= nib.load(input_dir+'sub-'+sub+'_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    data = extract_atlas_timeseries(sub, sub_dict, img_input_cleaned, file_atlas)
    print('shape of', sub, 'is', data.shape)
    np.save(folder+'/hcp_mmp1_'+sub+'.npy', data)
    
pool = Pool(4)
output = pool.map(main, sub_list)