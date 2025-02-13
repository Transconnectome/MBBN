from loss_writer import Writer
from learning_rate import LrHandler
from data_preprocess_and_load.dataloaders import DataHandler
import torch
import warnings
import numpy as np
from tqdm import tqdm
from model import *
import time
import pathlib
import os


from torch.nn import MSELoss,L1Loss,BCELoss, BCEWithLogitsLoss, Sigmoid


#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins

#torch AMP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# wandb
import wandb

# Import the time-series objects:
from nitime.timeseries import TimeSeries

# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer


class Trainer():
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self,sets,**kwargs):
        self.register_args(**kwargs)
        if self.target == 'depression':
            self.target = 'MDD_pp' # ABCD depression case!
        self.eval_iter = 0
        self.batch_index = None
        self.best_MAE = 1000
        self.best_loss = 100000
        self.best_AUROC = 0
        self.best_ACC = 0
        self.val_threshold = 0
        self.st_epoch = 1
        self.recent_pth = None
        self.state_dict = None
        self.transfer_learning =  bool(self.pretrained_model_weights_path) or self.finetune
        # weightwatcher stuffs
        self.weightwatcher = kwargs.get('weightwatcher')
        self.pretrained_model_weights_path = kwargs.get('pretrained_model_weights_path')
        self.exp_name = kwargs.get('exp_name')
        self.weightwatcher_save_dir = kwargs.get('weightwatcher_save_dir')
        
        if not self.weightwatcher:
            if self.fine_tune_task == 'regression':
                self.train_loader, self.val_loader, self.test_loader, self.mean, self.std = DataHandler(**kwargs).create_dataloaders()
            else:
                self.train_loader, self.val_loader, self.test_loader = DataHandler(**kwargs).create_dataloaders()
            
        self.lr_handler = LrHandler(self.train_loader, **kwargs)
                
        self.create_model() # model on cpu
        self.load_model_checkpoint()
        self.set_model_device() # set DDP or DP after loading checkpoint at CPUs
        
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)
        self.scaler = GradScaler() 
        
        
        self.load_optim_checkpoint()

        self.writer = Writer(sets, self.val_threshold, **kwargs)
        self.sets = sets
        
        
        
        #wandb
        os.environ["WANDB_API_KEY"] = self.wandb_key
        os.environ["WANDB_MODE"] = self.wandb_mode
        wandb.init(project=self.wandb_project,entity=self.wandb_entity,reinit=True, name=self.experiment_title, config=kwargs)
        wandb.watch(self.model,log='all',log_freq=10)
        
        self.nan_list = []

        for name, loss_dict in self.writer.losses.items():
            if loss_dict['is_active']:
                print('using {} loss'.format(name))
                setattr(self, name + '_loss_func', loss_dict['criterion'])

    
    def _sort_pth_files(self, files_Path):
        file_name_and_time_lst = []
        for f_name in os.listdir(files_Path):
            if f_name.endswith('.pth'):
                written_time = os.path.getctime(os.path.join(files_Path,f_name))
                file_name_and_time_lst.append((f_name, written_time))
        # Backward order of file creation time
        sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

        return sorted_file_lst
    
    def load_model_checkpoint(self):
        pths = self._sort_pth_files(self.experiment_folder)
        if self.transfer_learning:
            print(f'loading checkpoint from {self.pretrained_model_weights_path}')
            self.state_dict = torch.load(self.pretrained_model_weights_path, map_location='cpu') #, map_location=self.device
            self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=False)
            self.model.loaded_model_weights_path = self.pretrained_model_weights_path
        else:   
            if len(pths) > 0 : # if there are any checkpoints from which we can resume the training. 
                self.recent_pth = pths[0][0] # the most recent checkpoints
                print(f'loading checkpoint from {os.path.join(self.experiment_folder,self.recent_pth)}')
                self.state_dict = torch.load(os.path.join(self.experiment_folder,self.recent_pth),map_location='cpu') #, map_location=self.device
                if self.transfer_learning:
                    self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=False)
                else:    
                    self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=True)
                self.model.loaded_model_weights_path = os.path.join(self.experiment_folder,self.recent_pth)

            elif self.loaded_model_weights_path: # if there are weights from previous phase
                self.recent_pth = None
                self.state_dict = torch.load(self.loaded_model_weights_path,map_location='cpu') #, map_location=self.device
                self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=True)
                self.model.loaded_model_weights_path = self.loaded_model_weights_path

            else:
                self.recent_pth = None
                self.state_dict = None
                print('There are no checkpoints or weights from previous steps')
            
    def load_optim_checkpoint(self):
        if self.recent_pth: # if there are any checkpoints from which we can resume the training. 
            self.optimizer.load_state_dict(self.state_dict['optimizer_state_dict'])
            self.lr_handler.schedule.load_state_dict(self.state_dict['schedule_state_dict'])
            # self.optimizer.param_groups[0]['lr'] = self.state_dict['lr']
            self.scaler.load_state_dict(self.state_dict['amp_state'])
            self.st_epoch = int(self.state_dict['epoch']) + 1
            self.best_loss = self.state_dict['loss_value']
            text = 'Training start from epoch {} and learning rate {}.'.format(self.st_epoch, self.optimizer.param_groups[0]['lr'])
            if 'val_AUROC' in self.state_dict:
                text += 'validation AUROC - {} '.format(self.state_dict['val_AUROC'])
            print(text)
            
        elif self.state_dict:  # if there are weights from previous phase
            if self.transfer_learning:
                self.loaded_model_weights_path = self.pretrained_model_weights_path
            text = 'loaded model weights:\nmodel location - {}\nlast learning rate - {}\nvalidation loss - {}\n'.format(
                self.loaded_model_weights_path, self.state_dict['lr'],self.state_dict['loss_value'])
            if 'val_AUROC' in self.state_dict:
                text += 'validation AUROC - {}'.format(self.state_dict['val_AUROC'])
            
            if 'val_threshold' in self.state_dict:
                self.val_threshold = self.state_dict['val_threshold']
                text += 'val_threshold - {}'.format(self.state_dict['val_threshold'])
            print(text)
        else:
            pass

            
    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        params = self.model.parameters()
        weight_decay = self.kwargs.get('weight_decay')
        optim = self.kwargs.get('optim') # we use Adam or AdamW
        
        self.optimizer = getattr(torch.optim,optim)(params, lr=lr, weight_decay=weight_decay)
        
        
    def create_model(self):
        print('self.task:', self.task)
        if self.task.lower() == 'test':
            if self.fmri_type in ['timeseries','frequency', 'time_domain_low', 'time_domain_ultralow', 'time_domain_high', 'frequency_domain_low', 'frequency_domain_ultralow']:
                self.model = Transformer_Finetune(**self.kwargs)
            elif self.fmri_type == 'divided_timeseries':
                if self.fmri_dividing_type == 'three_channels':
                    self.model = Transformer_Finetune_Three_Channels(**self.kwargs)
                elif self.fmri_dividing_type == 'two_channels':
                    self.model = Transformer_Finetune_Two_Channels(**self.kwargs)

        elif self.task.lower() == 'vanilla_bert':
            self.model = Transformer_Finetune(**self.kwargs)

        elif self.task.lower() == 'mbbn':
            if self.fmri_dividing_type == 'three_channels':
                self.model = Transformer_Finetune_Three_Channels(**self.kwargs)
         
        elif self.task.lower() == 'divfreqbert_reconstruction':
            self.model = Transformer_Reconstruction_Three_Channels (**self.kwargs)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters of the model: {total_params}")
        
    def set_model_device(self):
        if self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                print('id of gpu is:', self.gpu)
                self.device = torch.device('cuda:{}'.format(self.gpu))
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True) 
                net_without_ddp = self.model.module
            else:
                self.device = torch.device("cuda" if self.cuda else "cpu")
                self.model.cuda()
                if 'reconstruction' in self.task.lower():
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model) 
                else: # having unused parameter (classifier token)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True) 
                model_without_ddp = self.model.module
        else:
            self.device = torch.device("cuda" if self.cuda else "cpu")
            
            self.model = DataParallel(self.model).to(self.device)

            


    def training(self):
        if self.weightwatcher:
            import weightwatcher
            import matplotlib.pyplot as plt
            import pandas as pd
            import sys
            watcher = weightwatcher.WeightWatcher(model=self.model)
            details = watcher.analyze()
            df = pd.DataFrame(details)
            df.to_csv(f'{self.weightwatcher_save_dir}/{self.exp_name}.csv', index=False) 
            plt.xlabel('layers')
            plt.ylabel('alpha')
            if int(self.exp_name.split('epoch')[-1]) < 100:
                if self.exp_name.split('_')[0] == 'HCPMMP1':
                    plt.ylim(0, 12)  
                    plt.yticks(range(0, 13, 2))  
                elif self.exp_name.split('_')[0] == 'Schaefer':
                    plt.ylim(0, 26) 
                    plt.yticks(range(0, 27, 2))  
            else:
                plt.ylim(0, 10)  
                plt.yticks(range(0, 11, 2))  
            plt.axhline(y=6, linestyle='--', color='r')
            plt.scatter(range(0, len(details)), details['alpha'])
            plt.savefig(f'{self.weightwatcher_save_dir}/{self.exp_name}.png', 
                dpi=600, 
                facecolor='w', 
                edgecolor='w',
                orientation='portrait', 
                format=None,
                transparent=False, 
                bbox_inches=None, 
                pad_inches=0.1,
                metadata=None)
            sys.exit()

        
        if self.profiling == True:
            self.nEpochs = 1
        for epoch in range(self.st_epoch,self.nEpochs + 1): 
            start = time.time()
            self.train_epoch(epoch)
            if self.target != 'reconstruction':
                if self.prepare_visualization:
                    self.eval_epoch('val')
                    print('\n______epoch summary {}/{}_____\n'.format(epoch,self.nEpochs))
                    self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
                    if self.fine_tune_task == 'regression':
                        self.writer.accuracy_summary(mid_epoch=False, mean=self.mean, std=self.std)
                    else:
                        self.writer.accuracy_summary(mid_epoch=False, mean=None, std=None)
                    self.writer.save_history_to_csv()

                    #wandb
                    if self.rank == 0:
                        self.writer.register_wandb(epoch, lr=self.optimizer.param_groups[0]['lr'])
                        self.save_checkpoint_(epoch, len(self.train_loader), self.scaler)
                    
                else:
                    self.eval_epoch('val')
                    self.eval_epoch('test')

                    print('\n______epoch summary {}/{}_____\n'.format(epoch,self.nEpochs))
                    self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
                    if self.fine_tune_task == 'regression':
                        self.writer.accuracy_summary(mid_epoch=False, mean=self.mean, std=self.std)
                    else:
                        self.writer.accuracy_summary(mid_epoch=False, mean=None, std=None)
                    self.writer.save_history_to_csv()

                    #wandb
                    if self.rank == 0:
                        self.writer.register_wandb(epoch, lr=self.optimizer.param_groups[0]['lr'])


                    for metric_name in dir(self.writer):
                        if 'history' not in metric_name:
                            continue
                        # metric_name =  save history to csv
                        metric_score = getattr(self.writer, metric_name)
            
                end = time.time()

                print(f'time taken to perform {epoch}: {end-start:.2f}')
            
            else:
                # reconstruction case #
                print('\n______epoch summary {}/{}_____\n'.format(epoch,self.nEpochs))
                self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
                if self.fine_tune_task == 'regression':
                    self.writer.accuracy_summary(mid_epoch=False, mean=self.mean, std=self.std)
                else:
                    self.writer.accuracy_summary(mid_epoch=False, mean=None, std=None)
                self.writer.save_history_to_csv()

                #wandb
                if self.rank == 0:
                    self.writer.register_wandb(epoch, lr=self.optimizer.param_groups[0]['lr'])
                    self.save_checkpoint_(epoch, len(self.train_loader), self.scaler) 

                end = time.time()

                print(f'time taken to perform {epoch}: {end-start:.2f}')
                
        return self.best_AUROC, self.best_loss, self.best_MAE #validation AUROC        
   
 
    def train_epoch(self,epoch):
        #torch.autograd.set_detect_anomaly(True)
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)
        self.train()
        
        if torch.cuda.is_available():
            max_allocated = torch.cuda.max_memory_allocated() 
            max_cached = torch.cuda.max_memory_cached() 

            print(f"max allocated memory: {max_allocated / (1024 * 1024):.2f} MB")
            print(f"max cached memory: {max_cached / (1024 * 1024):.2f} MB")
        else:
            print("no cuda device")

        times = []
        for batch_idx, input_dict in enumerate(tqdm(self.train_loader,position=0,leave=True)): 
            ### training ###
            #start_time = time.time()
            torch.cuda.nvtx.range_push("training steps")
            self.writer.total_train_steps += 1
            self.optimizer.zero_grad()
            if self.amp:
                torch.cuda.nvtx.range_push("forward pass")
                with autocast():
                    loss_dict, loss = self.forward_pass(input_dict)
                torch.cuda.nvtx.range_pop()
                loss = loss / self.accumulation_steps # gradient accumulation
                torch.cuda.nvtx.range_push("backward pass")
                self.scaler.scale(loss).backward()
                torch.cuda.nvtx.range_pop()

                if  (batch_idx + 1) % self.accumulation_steps == 0: # gradient accumulation
                    # gradient clipping 
                    if self.gradient_clipping == True:
                        self.scaler.unscale_(self.optimizer)
                        # print('executing gradient clipping')
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)

                    self.scaler.step(self.optimizer)
                    scale = self.scaler.get_scale()
                    self.scaler.update()
                    skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.lr_handler.schedule_check_and_update(self.optimizer) 
            else:
                loss_dict, loss = self.forward_pass(input_dict)
                loss.backward()
                self.optimizer.step()
                self.lr_handler.schedule_check_and_update(self.optimizer)
                
            self.writer.write_losses(loss_dict, set='train')
                
    def eval_epoch(self,set):
        loader = self.val_loader if set == 'val' else self.test_loader
        self.eval(set)
        with torch.no_grad():
            for batch_idx, input_dict in enumerate(tqdm(loader, position=0, leave=True)):
                with autocast():
                    loss_dict, _ = self.forward_pass(input_dict)
                self.writer.write_losses(loss_dict, set=set)
                if self.profiling == True:
                    if batch_idx == 10 : 
                        break
        
    def forward_pass(self,input_dict):
        input_dict = {k:(v.to(self.gpu) if (self.cuda and torch.is_tensor(v)) else v) for k,v in input_dict.items()}
        ###### test ######
        if self.task.lower() == 'test':
            if self.fmri_type in ['timeseries', 'frequency', 'time_domain_high', 'time_domain_low', 'time_domain_ultralow', 'frequency_domain_low', 'frequency_domain_ultralow', 'frequency_domain_high']:
                if self.ablation == 'raw_signal_for_att_conn':
                    output_dict = self.model(input_dict['fmri_sequence'], input_dict['fmri_sequence'], input_dict['fmri_sequence'])
                else:
                    output_dict = self.model(input_dict['fmri_sequence'])
            elif self.fmri_type == 'divided_timeseries':
                if self.fmri_dividing_type == 'two_channels':
                    output_dict = self.model(input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
                elif self.fmri_dividing_type == 'three_channels':
                    output_dict = self.model(input_dict['fmri_highfreq_sequence'], input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
  
        
        #### train & valid ####
        else:
            if self.fmri_type in ['timeseries', 'frequency', 'time_domain_high', 'time_domain_low', 'time_domain_ultralow', 'frequency_domain_low', 'frequency_domain_ultralow', 'frequency_domain_high']:
                if self.ablation == 'raw_signal_for_att_conn':
                    output_dict = self.model(input_dict['fmri_sequence'], input_dict['fmri_sequence'], input_dict['fmri_sequence'])
                else:
                    output_dict = self.model(input_dict['fmri_sequence'])
            elif self.fmri_type == 'divided_timeseries':
                if self.fmri_dividing_type == 'two_channels':
                    output_dict = self.model(input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
                elif self.fmri_dividing_type == 'three_channels':
                    output_dict = self.model(input_dict['fmri_highfreq_sequence'], input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
            
            
        torch.cuda.nvtx.range_push("aggregate_losses")
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        torch.cuda.nvtx.range_pop()
        if self.task.lower() in ['vanilla_bert', 'mbbn', 'test']:
            if self.target != 'reconstruction':
                self.compute_accuracy(input_dict, output_dict)
        return loss_dict, loss
    
    def aggregate_losses(self,input_dict,output_dict):
        final_loss_dict = {}
        final_loss_value = 0
        for loss_name, current_loss_dict in self.writer.losses.items():
            if current_loss_dict['is_active']:
                loss_func = getattr(self, 'compute_' + loss_name)
                torch.cuda.nvtx.range_push(f"{loss_name}")
                current_loss_value = loss_func(input_dict,output_dict)
                
                if current_loss_value.isnan().sum() > 0:
                    warnings.warn('found nans in computation')
                    print('at {} loss'.format(loss_name))
                    print(input_dict['subject_name'])
                    
                    if self.target != 'reconstruction':
                    
                        self.nan_list+=np.array(input_dict['subject_name'])[(output_dict[self.fine_tune_task].reshape(output_dict[self.fine_tune_task].shape[0],-1).isnan().sum(axis=1).detach().cpu().numpy() > 0)].tolist()
                        print('current_nan_list:',set(self.nan_list))
                    
                lamda = current_loss_dict['factor']
                factored_loss = current_loss_value * lamda
                final_loss_dict[loss_name] = factored_loss.item()
                final_loss_value += factored_loss
                
        
        final_loss_dict['total'] = final_loss_value.item()
        
        return final_loss_dict, final_loss_value


        
    def testing(self):
        self.eval_epoch('test')
        self.writer.loss_summary(lr=0)
        if self.fine_tune_task == 'regression':
            self.writer.accuracy_summary(mid_epoch=False, mean=self.mean, std=self.std)
        else:
            self.writer.accuracy_summary(mid_epoch=False, mean=None, std=None)
        for metric_name in dir(self.writer):
            if 'history' not in metric_name:
                continue
            metric_score = getattr(self.writer, metric_name)
    
    def train(self):
        self.mode = 'train'
        self.model = self.model.train()
        
    def eval(self,set):
        self.mode = set
        self.model = self.model.eval()

    def get_last_loss(self):
        if self.kwargs.get('target') == 'reconstruction':
            return self.writer.total_train_loss_history[-1]
        else:
            return self.writer.total_val_loss_history[-1]

    def get_last_AUROC(self):
        if hasattr(self.writer,'val_AUROC'):
            return self.writer.val_AUROC[-1]
        else:
            return None

    def get_last_MAE(self):
        if hasattr(self.writer,'val_MAE'):
            return self.writer.val_MAE[-1]
        else:
            return None
        
        
    def get_last_ACC(self):
        if hasattr(self.writer,'val_Balanced_Accuracy'):
            return self.writer.val_Balanced_Accuracy[-1]
        else:
            return None
    
    def get_last_best_ACC(self):
        if hasattr(self.writer,'val_best_bal_acc'):
            return self.writer.val_best_bal_acc[-1]
        else:
            return None
    
    def get_last_val_threshold(self):
        if hasattr(self.writer,'val_best_threshold'):
            return self.writer.val_best_threshold[-1]
        else:
            return None

    def save_checkpoint_(self, epoch, batch_idx, scaler):
        loss = self.get_last_loss()
        #accuracy = self.get_last_AUROC()
        val_ACC = self.get_last_ACC()
        val_best_ACC = self.get_last_best_ACC()
        val_AUROC = self.get_last_AUROC()
        val_MAE = self.get_last_MAE()
        val_threshold = self.get_last_val_threshold()
        title = str(self.writer.experiment_title) + '_epoch_' + str(int(epoch))
        directory = self.writer.experiment_folder

        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.amp:
            amp_state = scaler.state_dict()

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.model.module.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict() if self.optimizer is not None else None,
            'epoch':epoch,
            'loss_value':loss,
            'amp_state': amp_state}

        # if val_ACC is not None:
        #     ckpt_dict['val_ACC'] = val_ACC
        if val_AUROC is not None:
            ckpt_dict['val_AUROC'] = val_AUROC
        if val_threshold is not None:
            ckpt_dict['val_threshold'] = val_threshold
        if val_MAE is not None:
            ckpt_dict['val_MAE'] = val_MAE
        if self.lr_handler.schedule is not None:
            ckpt_dict['schedule_state_dict'] = self.lr_handler.schedule.state_dict()
            ckpt_dict['lr'] = self.optimizer.param_groups[0]['lr']
            print(f"current_lr:{self.optimizer.param_groups[0]['lr']}")
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        
        # classification
        if val_AUROC is not None:
            if self.best_AUROC < val_AUROC:
                self.best_AUROC = val_AUROC
                name = "{}_BEST_val_AUROC.pth".format(title)
                torch.save(ckpt_dict, os.path.join(directory, name))
                print(f'updating best saved model with AUROC:{val_AUROC}')

                if self.best_ACC < val_ACC:
                    self.best_ACC = val_ACC
            elif self.best_AUROC >= val_AUROC:
                # If model is not improved in val AUROC, but improved in val ACC.
                if self.best_ACC < val_ACC:
                    self.best_ACC = val_ACC
                    name = "{}_BEST_val_ACC.pth".format(title)
                    torch.save(ckpt_dict, os.path.join(directory, name))
                    print(f'updating best saved model with ACC:{val_ACC}')

        # regression
        elif val_AUROC is None and val_MAE is not None:
            if self.best_MAE > val_MAE:
                self.best_MAE = val_MAE
                name = "{}_BEST_val_MAE.pth".format(title)
                torch.save(ckpt_dict, os.path.join(directory, name))
                print(f'updating best saved model with MAE: {val_MAE}')
            else:
                pass
                
        else:
            if self.best_loss > loss:
                self.best_loss = loss
                name = "{}_BEST_val_loss.pth".format(title)
                torch.save(ckpt_dict, os.path.join(directory, name))
                print(f'updating best saved model with loss: {loss}')
            else:
                pass

    def compute_reconstruction(self,input_dict,output_dict):
                  
        fmri_highfreq_sequence = input_dict['fmri_highfreq_sequence']
        reconstruction_loss_high = self.reconstruction_loss_func(fmri_highfreq_sequence,
                                                                 output_dict['reconstructed_high_fmri_sequence'])          
                  
        fmri_lowfreq_sequence = input_dict['fmri_lowfreq_sequence']
        reconstruction_loss_low = self.reconstruction_loss_func(fmri_lowfreq_sequence,
                                                                 output_dict['reconstructed_low_fmri_sequence'])
        
        fmri_ultralowfreq_sequence = input_dict['fmri_ultralowfreq_sequence']
        reconstruction_loss_ultralow = self.reconstruction_loss_func(fmri_ultralowfreq_sequence,
                                                                     output_dict['reconstructed_ultralow_fmri_sequence'])
        
                  
        reconstruction_loss = reconstruction_loss_high + reconstruction_loss_low + reconstruction_loss_ultralow
                                                                
        return reconstruction_loss
    
    
    def compute_spatial_difference(self,input_dict,output_dict):
        spatial_difference_loss = self.spatial_difference_loss_func(output_dict['high_spatial_attention'], output_dict['low_spatial_attention'], output_dict['ultralow_spatial_attention'])
        return spatial_difference_loss
   
    def compute_mask(self, input_dict, output_dict):
        fmri_highfreq_sequence = input_dict['fmri_highfreq_sequence']
        fmri_lowfreq_sequence = input_dict['fmri_lowfreq_sequence']          
        fmri_ultralowfreq_sequence = input_dict['fmri_ultralowfreq_sequence']
                  
        if self.masking_method == 'temporal':
            if self.temporal_masking_type == 'single_point':  
                mask_loss_high = self.mask_loss_func(fmri_highfreq_sequence,
                                                    output_dict['mask_single_point_high_fmri_sequence'])

                mask_loss_low = self.mask_loss_func(fmri_lowfreq_sequence,
                                                    output_dict['mask_single_point_high_fmri_sequence'])

                mask_loss_ultralow = self.mask_loss_func(fmri_ultralowfreq_sequence,
                                                         output_dict['mask_single_point_high_fmri_sequence'])
            elif self.temporal_masking_type == 'time_window':  
                mask_loss_high = self.mask_loss_func(fmri_highfreq_sequence,
                                                    output_dict['mask_time_window_high_fmri_sequence'])

                mask_loss_low = self.mask_loss_func(fmri_lowfreq_sequence,
                                                    output_dict['mask_time_window_low_fmri_sequence'])

                mask_loss_ultralow = self.mask_loss_func(fmri_ultralowfreq_sequence,
                                                         output_dict['mask_time_window_ultralow_fmri_sequence'])
        elif self.masking_method == 'spatial':
            mask_loss_high = self.mask_loss_func(fmri_highfreq_sequence,
                                                 output_dict['mask_hub_ROIs_high_fmri_sequence'])

            mask_loss_low = self.mask_loss_func(fmri_lowfreq_sequence,
                                                output_dict['mask_hub_ROIs_low_fmri_sequence'])

            mask_loss_ultralow = self.mask_loss_func(fmri_ultralowfreq_sequence,
                                                     output_dict['mask_hub_ROIs_ultralow_fmri_sequence'])    
                  
        else: # spatiotemporal
            if self.spatiotemporal_masking_type == 'separate':
                temporal_mask_loss_high = self.mask_loss_func(fmri_highfreq_sequence,
                                                              output_dict['temporal_mask_spatiotemporal_high_fmri_sequence'])
                spatial_mask_loss_high = self.mask_loss_func(fmri_highfreq_sequence,
                                                              output_dict['spatial_mask_spatiotemporal_high_fmri_sequence'])
                mask_loss_high = temporal_mask_loss_high + spatial_mask_loss_high
                  
                temporal_mask_loss_low = self.mask_loss_func(fmri_lowfreq_sequence,
                                                              output_dict['temporal_mask_spatiotemporal_low_fmri_sequence'])
                spatial_mask_loss_low = self.mask_loss_func(fmri_lowfreq_sequence,
                                                              output_dict['spatial_mask_spatiotemporal_low_fmri_sequence'])
                mask_loss_low = temporal_mask_loss_low + spatial_mask_loss_low
                  
                temporal_mask_loss_ultralow = self.mask_loss_func(fmri_ultralowfreq_sequence,
                                                              output_dict['temporal_mask_spatiotemporal_ultralow_fmri_sequence'])
                spatial_mask_loss_ultralow = self.mask_loss_func(fmri_ultralowfreq_sequence,
                                                              output_dict['spatial_mask_spatiotemporal_ultralow_fmri_sequence'])
                mask_loss_ultralow = temporal_mask_loss_ultralow + spatial_mask_loss_ultralow
                  
            else:
                mask_loss_high = self.mask_loss_func(fmri_highfreq_sequence,
                                                     output_dict['mask_spatiotemporal_high_fmri_sequence'])

                mask_loss_low = self.mask_loss_func(fmri_lowfreq_sequence,
                                                    output_dict['mask_spatiotemporal_low_fmri_sequence'])

                mask_loss_ultralow = self.mask_loss_func(fmri_ultralowfreq_sequence,
                                                         output_dict['mask_spatiotemporal_ultralow_fmri_sequence'])       
                  
        mask_loss = mask_loss_high + mask_loss_low + mask_loss_ultralow
        
                  
        return mask_loss
        
        
    def compute_binary_classification(self,input_dict,output_dict):
        binary_loss = self.binary_classification_loss_func(output_dict['binary_classification'].squeeze(), input_dict[self.target].squeeze().float()) # BCEWithLogitsLoss
        if torch.sum(torch.isnan(binary_loss)):
            binary_loss = torch.nan_to_num(binary_loss, nan=0.0)
        return binary_loss

    def compute_regression(self,input_dict,output_dict):
        # normalized target, normalized logits to original scale
        
        regression_loss = self.regression_loss_func(output_dict['regression'].squeeze(), input_dict[self.target].squeeze()) #self.regression_loss_func(output_dict['regression'].squeeze(),input_dict['subject_regression'].squeeze())
        return regression_loss

    def compute_accuracy(self,input_dict,output_dict):
        task = self.kwargs.get('fine_tune_task') #self.model.task
        out = output_dict[task].detach().clone().cpu()
        score = out.squeeze() if out.shape[0] > 1 else out
        labels = input_dict[self.target].clone().cpu() # input_dict['subject_' + task].clone().cpu()
        subjects = input_dict['subject'].clone().cpu()
        for i, subj in enumerate(subjects):
            subject = str(subj.item())
            if subject not in self.writer.subject_accuracy:
                self.writer.subject_accuracy[subject] = {'score': score[i].unsqueeze(0), 'mode': self.mode, 'truth': labels[i],'count': 1}
            else:
                self.writer.subject_accuracy[subject]['score'] = torch.cat([self.writer.subject_accuracy[subject]['score'], score[i].unsqueeze(0)], dim=0)
                self.writer.subject_accuracy[subject]['count'] += 1

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
