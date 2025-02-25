import numpy as np
import torch
from torch.utils.data import DataLoader,Subset, Dataset, RandomSampler
from pathlib import Path
#DDP
from torch.utils.data.distributed import DistributedSampler
from data_preprocess_and_load.datasets import *
from utils import reproducibility
import os
import nibabel as nib
from sklearn.model_selection import train_test_split
import pandas as pd


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
            self.target == 'DX_GROUP'
        elif self.target == 'nihtbx_totalcomp_uncorrected':
            self.target = 'total_intelligence'
        elif self.target == 'ASD_label':
            if self.dataset_name == 'ABCD':
                self.target = 'ASD'
        

        self.splits_folder = Path(self.base_path).joinpath('splits',self.dataset_name)
        self.current_split = self.splits_folder.joinpath(f"{self.dataset_name}_{self.target}_ROI_{self.intermediate_vec}_seq_len_{self.seq_len}_split{self.seed}.txt")
        
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
                    metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABCD_phenotype_total.csv'))
            elif self.dataset_name == 'UKB':
                sub = [str(i) for i in os.listdir(kwargs.get('ukb_path'))]
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
                    if self.intermediate_vec == 180:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'hcp_mmp1_180_sub-'+i+'.npy')
                    elif self.intermediate_vec == 360:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'hcp_mmp1_sub-'+i+'.npy')
                    elif self.intermediate_vec == 400:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'schaefer_sub-'+i+'.npy')
                    file = np.load(filename)[:self.seq_len].T
                elif self.dataset_name == 'UKB':
                    if self.intermediate_vec == 180:
                        filename = os.path.join(kwargs.get('ukb_path'), i+'/'+'hcp_mmp1_'+i+'.npy')
                    elif self.intermediate_vec == 400:
                        filename = os.path.join(kwargs.get('ukb_path'), i+'/'+'schaefer_400Parcels_17Networks_'+i+'.npy')
                    file = np.load(filename)[20:20+self.seq_len].T
                elif self.dataset_name == 'ABIDE':
                    ## only have schaefer atlas due to the /storage problem .. :(
                    filename = os.path.join(kwargs.get('abide_path'), i+'/'+'schaefer_400Parcels_17Networks_'+i+'.npy')
                    file = np.load(filename)[20:20+self.seq_len].T
                    
                 
                if file.shape[1] >= self.seq_len:
                    valid_sub.append(i)
                    for j in range(file.shape[0]):
                        if np.sum(file[j]) == 0:
                            prob_sub.append(i)

                            
            valid_sub = list(set(valid_sub) - set(prob_sub))
                
            if self.dataset_name == 'UKB':
                valid_sub = list(map(int, valid_sub))
                        
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
                        if self.fine_tune_task == 'binary_classification':
                            X_train, X_test, y_train, y_test = train_test_split(new_meta['subjectkey'],
                                                                    new_meta[self.target],
                                                                    test_size=0.15,
                                                                    stratify= new_meta[self.target],
                                                                    random_state = self.seed)

                            X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.175,
                                                                              stratify= y_train,
                                                                              random_state = self.seed)
                        elif self.fine_tune_task == 'regression':
                            X_train, X_test, y_train, y_test = train_test_split(new_meta['subjectkey'],
                                                                    new_meta[self.target],
                                                                    test_size=0.15,
                                                                    random_state = self.seed)

                            X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.175,
                                                                              random_state = self.seed)
                            
                            
                elif self.dataset_name == 'UKB':
                    valid_df = pd.DataFrame(valid_sub).rename(columns = {0 : 'eid'})
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='eid')
                    if self.fine_tune_task == 'binary_classification':
                        X_train, X_test, y_train, y_test = train_test_split(new_meta['eid'],
                                                                    new_meta[self.target],
                                                                    test_size=0.15,
                                                                    stratify= new_meta[self.target],
                                                                    random_state = self.seed)

                        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.175,
                                                                              stratify= y_train,
                                                                              random_state = self.seed)
                    elif self.fine_tune_task == 'regression':
                        X_train, X_test, y_train, y_test = train_test_split(new_meta['eid'],
                                                                new_meta[self.target],
                                                                test_size=0.15,
                                                                random_state = self.seed)

                        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                          y_train,
                                                                          test_size=0.175,
                                                                          random_state = self.seed)
                elif self.dataset_name == 'ABIDE':
                    
                    subid = valid_sub # [i[2:] if i.startswith('00') else i for i in valid_sub]
                    # starts with 5 or 2 now! (because metadata doesn't starts with 00)

                    valid_df = pd.DataFrame(subid).rename(columns = {0 : 'SUB_ID'}) # ['SUB_ID']
                    new_meta = new_meta.rename(columns= {'DX_GROUP': 'ASD'}) # ['SUB_ID', 'ASD'] # 두 dataframe 다 int.

                    new_meta['SUB_ID'] = new_meta['SUB_ID'].astype(str)
                    new_meta['SUB_ID'] = ['00' + i if '00' + i in subid else i for i in new_meta['SUB_ID']]
                    # now new_meta's SUB_ID contains 00
                    
                    valid_df['SUB_ID'] = valid_df['SUB_ID'].astype(str)
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='SUB_ID')
                    
                    if self.target == 'DX_GROUP':
                        self.target = 'ASD'
                    X_train, X_test, y_train, y_test = train_test_split(new_meta['SUB_ID'],
                                                                new_meta[self.target],
                                                                test_size=0.15,
                                                                stratify= new_meta[self.target],
                                                                random_state = self.seed)

                    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                          y_train,
                                                                          test_size=0.175,
                                                                          stratify= y_train,
                                                                          random_state = self.seed)
                    
                
                
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
            with open(f"./splits/{self.dataset_name}/{self.dataset_name}_{self.target}_ROI_{self.intermediate_vec}_seq_len_{self.seq_len}_split{self.seed}.txt", mode="w") as file:
                file.write('\n'.join(sublist))
        print(self.current_split)
        
    def get_mean_std(self):
        return None
    
    def get_dataset(self):
        if self.dataset_name == 'ABCD':
            return ABCD_fMRI_timeseries
        elif self.dataset_name == 'HCP1200':
            return HCP_fMRI_timeseries
        elif self.dataset_name == 'ABIDE':
            return ABIDE_fMRI_timeseries
        elif self.dataset_name == 'UKB':
            return UKB_fMRI_timeseries
        
    def current_split_exists(self):
        return self.current_split.exists() 


    def create_dataloaders(self):
        reproducibility(**self.kwargs) 

        subject = open(self.current_split, 'r').readlines()
        subject = [x[:-1] for x in subject]
        subject.remove('train_subjects')
        subject.remove('val_subjects')
        subject.remove('test_subjects')
        self.subject_list = self.train_dataset.index_l
        
        
        if self.current_split_exists():
            print('loading splits')
            train_names, val_names, test_names = self.load_split()
            train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(train_names,val_names,test_names,self.subject_list)
        

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
                valid_sampler = DistributedSampler(val_dataset, shuffle=False)
                print('length of valid sampler is:', len(valid_sampler)) # 5
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
                print('length of test sampler is:', len(test_sampler))
        else:
            train_sampler = RandomSampler(train_dataset)
            if self.target != 'reconstruction':
                valid_sampler = RandomSampler(val_dataset)
                test_sampler = RandomSampler(test_dataset)
        
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
        #if eval:
        #    workers = 0
        
        def worker_init_fn(worker_id):
            torch.cuda.empty_cache()  # 각 worker 시작 시 GPU 캐시 정리
        
        params = {'batch_size': batch_size,
                  #'shuffle': True,
                  'num_workers': workers,
                  'drop_last': True,
                  'pin_memory': True,  # True if cuda else False,
                  'persistent_workers': True if workers > 0 and cuda else False,
                  'prefetch_factor' : 2 if workers>0 else None,
                  'worker_init_fn' : worker_init_fn}
        '''
        params = {'batch_size': batch_size,
          #'shuffle': True,
          'num_workers': workers,
          'drop_last': True,
          'pin_memory': False,
          'persistent_workers': False}
        '''
        return params

    def convert_subject_list_to_idx_list(self,train_names,val_names,test_names,subj_list):
        subj_idx = np.array([str(x[1]) for x in subj_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        
        return train_idx,val_idx,test_idx
    
    def load_split(self):
        subject_order = open(self.current_split, 'r').readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(['train' in line for line in subject_order])
        val_index = np.argmax(['val' in line for line in subject_order])
        test_index = np.argmax(['test' in line for line in subject_order])
        train_names = subject_order[train_index + 1:val_index] # NDAR~ 형태
        val_names = subject_order[val_index+1:test_index]
        test_names = subject_order[test_index + 1:]
              
        return train_names,val_names,test_names
