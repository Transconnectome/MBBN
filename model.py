import os
from abc import ABC, abstractmethod
import scipy
import torch
from transformers import BertConfig,BertPreTrainedModel, BertModel
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Attention(nn.Module):
    '''
    N = ROIs, C = sequence length
    '''
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)

    def batch_to_head_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def head_to_batch_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor
     
    def forward(self, x, return_attn=True):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # q: B, num_heads, N, C // num_heads
        # k: B, num_heads, N, C // num_heads
        # v: B, num_heads, N, C // num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn : batch, num_heads, ROI, ROI

        if return_attn:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Classifier(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.best_loss = 1000000
        #self.best_accuracy = 0
        self.best_AUROC = 0        
        
    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device
    
    def register_vars(self,**kwargs):
        self.intermediate_vec = kwargs.get('intermediate_vec') # embedding size(h) 
        self.spatiotemporal = kwargs.get('spatiotemporal')
        self.spatial = kwargs.get('spatial')
        self.temporal = kwargs.get('temporal')
        self.transformer_dropout_rate = kwargs.get('transformer_dropout_rate')
        self.sequence_length = kwargs.get('sequence_length')
        self.pretrained_model_weights_path = kwargs.get('pretrained_model_weights_path')
        self.finetune = kwargs.get('finetune')
        self.transfer_learning =  bool(self.pretrained_model_weights_path) or self.finetune
        self.finetune_test = kwargs.get('finetune_test') # test phase of finetuning task
        self.num_heads = kwargs.get('num_heads')
        self.target = kwargs.get('target')
        self.task = kwargs.get('fine_tune_task')
        self.step = kwargs.get('step')
        self.visualization = kwargs.get('visualization')
        self.ablation = kwargs.get('ablation') 
        
        if self.transfer_learning or self.finetune_test:
            self.sequence_length += (464-self.sequence_length)
            
        if kwargs.get('fmri_type') == 'divided_timeseries':
            self.BertConfig = BertConfig(hidden_size=self.intermediate_vec, vocab_size=1,
                             num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                             num_attention_heads=self.num_heads, max_position_embeddings=self.sequence_length+1,
                             hidden_dropout_prob=self.transformer_dropout_rate)
        else:
            self.BertConfig = BertConfig(hidden_size=self.intermediate_vec, vocab_size=1,
                         num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                         num_attention_heads=self.num_heads, max_position_embeddings=self.sequence_length+1,
                         hidden_dropout_prob=self.transformer_dropout_rate)
                
        self.label_num = 1
        self.use_cuda = kwargs.get('gpu') #'cuda'
        self.dataset_name = kwargs.get('dataset_name')

    def load_partial_state_dict(self, state_dict,load_cls_embedding):
        print('loading parameters onto new model...')
        own_state = self.state_dict()
        loaded = {name:False for name in own_state.keys()}
        for name, param in state_dict.items():
            if name not in own_state:
                print('notice: {} is not part of new model and was not loaded.'.format(name))
                continue
            elif 'cls_embedding' in name and not load_cls_embedding:
                continue
            elif 'position' in name and param.shape != own_state[name].shape:
                print('debug line above')
                continue
            param = param.data
            own_state[name].copy_(param)
            loaded[name] = True
        for name,was_loaded in loaded.items():
            if not was_loaded:
                print('notice: named parameter - {} is randomly initialized'.format(name))
                # not in state dict but in model..

    def save_checkpoint(self, directory, title, epoch, loss, AUROC, optimizer=None,schedule=None):
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.state_dict(),
            'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
            'epoch':epoch,
            'loss_value':loss}
        if AUROC is not None:
            ckpt_dict['AUROC'] = AUROC
        if schedule is not None:
            ckpt_dict['schedule_state_dict'] = schedule.state_dict()
            ckpt_dict['lr'] = schedule.get_last_lr()[0]
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        
        # save model with last epoch
        core_name = title
        name = "{}_last_epoch.pth".format(core_name)
        torch.save(ckpt_dict, os.path.join(directory, name)) 
        
        # save model with best loss or best AUROC/ACC
        if AUROC is None and self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST_val_loss.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')
        if AUROC is not None and self.best_AUROC < AUROC:
            self.best_AUROC = AUROC
            name = "{}_BEST_val_AUROC.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')
    
class Transformer_Block(BertPreTrainedModel, BaseModel):    
    def __init__(self,config,**kwargs):
        super(Transformer_Block, self).__init__(config)
        self.register_vars(**kwargs)
        self.cls_pooling = True
        self.init_weights()
        self.bert = BertModel(config, add_pooling_layer=self.cls_pooling)        
        self.cls_embedding = nn.Sequential(nn.Linear(self.intermediate_vec, self.intermediate_vec), nn.LeakyReLU())
        self.register_buffer('cls_id', (torch.ones((1, 1, self.intermediate_vec)) * 0.5), persistent=False)
        
                
    def concatenate_cls(self, x):
        cls_token = self.cls_embedding(self.cls_id.expand(x.size()[0], -1, -1))
        return torch.cat([cls_token, x], dim=1)
    
    def forward(self, x):
        inputs_embeds = self.concatenate_cls(x) # (batch, seq_len+1, ROI)
        outputs = self.bert(input_ids=None,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=inputs_embeds, #give our embeddings
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=True
                            )

        sequence_output = outputs[0][:, 1:, :] # (batch, seq_len, ROI) - for recon loss
        # last hidden state : Sequence of hidden-states at the output of the last layer of the model.
        
        pooled_cls = outputs[1] # (batch, ROI) - for output
        # pooler output : Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        
        return {'sequence': sequence_output, 'cls': pooled_cls}

# ablation study 1 - no frequency dividing
class Transformer_Finetune(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Finetune, self).__init__()
        self.register_vars(**kwargs)
        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num) #.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
         # x is (batch size, seq len, ROI)
        
        transformer_dict = self.transformer(x)
        
        '''
        size of out seq is: (batch, seq_len, ROI)
        size of out cls is: (batch, ROI)
        size of prediction is: (batch, label_num)
        '''
        out_seq = transformer_dict['sequence']
        out_cls = transformer_dict['cls']
        prediction = self.regression_head(out_cls)
        
        return {self.task:prediction}
    
# ablation study 2 - two frequency version (no high freq)
class Transformer_Finetune_Two_Channels(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Finetune_Two_Channels, self).__init__()
        self.register_vars(**kwargs)
        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        
        if self.sequence_length % 12 == 0:
            num_heads = 12 # 36
        elif self.sequence_length % 8 == 0:
            num_heads = 8

        self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        self.regression_head = Classifier(self.intermediate_vec, self.label_num)

    def forward(self, x_l, x_u):        
        # 01 model
        transformer_dict_low = self.transformer(x_l)
        transformer_dict_ultralow = self.transformer(x_u)

        low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1)) # (batch, ROI, seq_len)
        ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1)) # (batch, ROI, seq_len)

        # 02 get pooled_cls
        out_cls_low = transformer_dict_low['cls']
        out_cls_ultralow = transformer_dict_ultralow['cls']

        pred_low = self.regression_head(out_cls_low)
        pred_ultralow = self.regression_head(out_cls_ultralow)
      
        prediction = (pred_low+pred_ultralow)/2

        ans_dict = {self.task:prediction, 'low_spatial_attention':low_spatial_attention, 'ultralow_spatial_attention':ultralow_spatial_attention}

        return ans_dict   
    
# ablation study 3 - convolution
## main model ##
class Transformer_Finetune_Three_Channels(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Finetune_Three_Channels, self).__init__()
        self.register_vars(**kwargs)

        if self.ablation == 'convolution':
            self.cnn = nn.Conv1d(self.sequence_length, self.sequence_length, 3, stride=1, padding=1)            
                
        if self.spatiotemporal:
            # temporal
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
            
            # spatial
            if self.sequence_length % 12 == 0:
                num_heads = 12 # 36
            elif self.sequence_length % 8 == 0:
                num_heads = 8
            self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            
            # classifier
            self.regression_head = Classifier(self.intermediate_vec, self.label_num)
            
        elif self.temporal:
            # temporal
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
            
            # classifier
            self.regression_head = Classifier(self.intermediate_vec, self.label_num)
            
        elif self.spatial:
            # spatial
            if self.sequence_length % 12 == 0:
                num_heads = 12 # 36
            elif self.sequence_length % 8 == 0:
                num_heads = 8
            self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        
            # classifier
            self.regression_head = Classifier(self.intermediate_vec*self.intermediate_vec, self.label_num)
        
            
    def forward(self, x_h, x_l, x_u):
        # input shape : (batch, seq_len, ROI)
        device = x_h.get_device()

        if self.ablation == 'convolution':
            x_h = self.cnn(x_h)
            x_l = self.cnn(x_l)
            x_u = self.cnn(x_u)

        # 01 get dict
        if self.spatiotemporal:       
            # temporal
            transformer_dict_high = self.transformer(x_h)
            transformer_dict_low = self.transformer(x_l)
            transformer_dict_ultralow = self.transformer(x_u)
            # spatial
            high_spatial_attention = self.high_spatial_attention(x_h.permute(0, 2, 1)) # (batch, ROI, sequence length)
            low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1)) # (batch, ROI, sequence length)
            ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1)) # (batch, ROI, sequence length)
            # desired output shape : (batch, num_heads, ROI, ROI)
        
            
        elif self.temporal:
            transformer_dict_high = self.transformer(x_h)
            transformer_dict_low = self.transformer(x_l)
            transformer_dict_ultralow = self.transformer(x_u)
        
        elif self.spatial:
            high_spatial_attention = self.high_spatial_attention(x_h.permute(0, 2, 1)) # (batch, ROI, sequence length)
            low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1)) # (batch, ROI, sequence length)
            ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1)) # (batch, ROI, sequence length)
            # desired output shape : (batch, num_heads, ROI, ROI)

        # 02 get pooled_cls
        if not self.spatial:
            out_cls_high = transformer_dict_high['cls']
            out_cls_low = transformer_dict_low['cls']
            out_cls_ultralow = transformer_dict_ultralow['cls']
        else:
            batch_size = high_spatial_attention.shape[0]
            out_cls_high = torch.mean(high_spatial_attention, dim=1).reshape(batch_size, -1)
            out_cls_low = torch.mean(low_spatial_attention, dim=1).reshape(batch_size, -1)
            out_cls_ultralow = torch.mean(ultralow_spatial_attention, dim=1).reshape(batch_size, -1)
            
        pred_high = self.regression_head(out_cls_high)
        pred_low = self.regression_head(out_cls_low)
        pred_ultralow = self.regression_head(out_cls_ultralow)
            

        prediction = (pred_high+pred_low+pred_ultralow)/3

        if self.visualization:
            ans_dict = prediction
        else:
            if self.spatiotemporal or self.spatial:
                ans_dict = {self.task:prediction, 'high_spatial_attention':high_spatial_attention, 'low_spatial_attention':low_spatial_attention, 'ultralow_spatial_attention':ultralow_spatial_attention}
            else:
                ans_dict = {self.task:prediction}

        return ans_dict
    
class Transformer_Reconstruction_Three_Channels(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Reconstruction_Three_Channels, self).__init__()

        # mask loss
        self.mask_loss = kwargs.get('use_mask_loss')
        self.masking_method = kwargs.get('masking_method') # spatial temporal spatiotemporal

        ## temporal masking
        self.masking_rate = kwargs.get('masking_rate')
        self.temporal_masking_type = kwargs.get('temporal_masking_type') # single point, time window
        self.temporal_masking_window_size = kwargs.get('temporal_masking_window_size') 
        self.window_interval_rate = kwargs.get('window_interval_rate')
        
        ## spatial masking
        self.spatial_masking_type = kwargs.get('spatial_masking_type') # hub ROIs, random ROIs
        self.num_hub_ROIs = kwargs.get('num_hub_ROIs')
        self.num_random_ROIs = kwargs.get('num_random_ROIs')

        ## spatiotemporal masking
        self.spatiotemporal_masking_type = kwargs.get('spatiotemporal_masking_type')
        self.spatiotemporal = kwargs.get('spatiotemporal') # spatial loss
        self.communicability_option = kwargs.get('communicability_option')
        
        # recon loss
        self.recon_loss = kwargs.get('use_recon_loss')

        self.register_vars(**kwargs)
        
        if self.spatiotemporal:
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

            if self.sequence_length % 12 == 0:
                    num_heads = 12 # 36
            elif self.sequence_length % 8 == 0:
                num_heads = 8

            self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
                    
            
        else:
            ## temporal case
            self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
            
        
    def forward(self, x_h, x_l, x_u):
        ans_dict = {}
        
        if self.spatiotemporal:
            ## spatial loss ##
            high_spatial_attention = self.high_spatial_attention(x_h.permute(0, 2, 1)) # (batch, ROI, sequence length)
            low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1)) # (batch, ROI, sequence length)
            ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1)) # (batch, ROI, sequence length)
            # desired output shape : (batch, num_heads, ROI, ROI)
            
            ans_dict['high_spatial_attention'] = high_spatial_attention
            ans_dict['low_spatial_attention'] = low_spatial_attention
            ans_dict['ultralow_spatial_attention'] = ultralow_spatial_attention
            
        if self.mask_loss:
            if not (self.temporal_masking_type == 'spatiotemporal' and self.spatiotemporal_masking_type == 'separate'):
                masked_seq_high = x_h
                masked_seq_low = x_l
                masked_seq_ultralow = x_u
            batch_size = x_h.shape[0]
            if self.masking_method == 'temporal':
                if self.temporal_masking_type == 'single_point':
                    number = int(self.sequence_length * self.masking_rate)
                    mask_list = np.random.randint(0, self.sequence_length, size=number)
                    for mask in mask_list:
                        # generate masked sequence
                        masked_seq_high[:, mask:mask+1, :] = torch.zeros(batch_size, 1, self.intermediate_vec)
                        masked_seq_low[:, mask:mask+1, :] = torch.zeros(batch_size, 1, self.intermediate_vec)
                        masked_seq_ultralow[:, mask:mask+1, :] = torch.zeros(batch_size, 1, self.intermediate_vec) 

                    transformer_dict_high_mask = self.transformer(masked_seq_high)
                    mask_out_seq_high = transformer_dict_high_mask['sequence']    
                    transformer_dict_low_mask = self.transformer(masked_seq_low)
                    mask_out_seq_low = transformer_dict_low_mask['sequence']
                    transformer_dict_ultralow_mask = self.transformer(masked_seq_ultralow)
                    mask_out_seq_ultralow = transformer_dict_ultralow_mask['sequence']

                    ans_dict['mask_single_point_high_fmri_sequence'] = mask_out_seq_high
                    ans_dict['mask_single_point_low_fmri_sequence'] = mask_out_seq_low
                    ans_dict['mask_single_point_ultralow_fmri_sequence'] = mask_out_seq_ultralow


                if self.temporal_masking_type == 'time_window':
                    mask_list = list(range(0, self.sequence_length, self.window_interval_rate*self.temporal_masking_window_size))
                    if self.sequence_length - mask_list[-1] < self.temporal_masking_window_size:
                        mask_list = mask_list[:-1]

                    for mask in mask_list:
                        masked_seq_high[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)
                        masked_seq_low[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)
                        masked_seq_ultralow[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)

                    transformer_dict_high_mask = self.transformer(masked_seq_high)
                    mask_out_seq_high = transformer_dict_high_mask['sequence']    
                    transformer_dict_low_mask = self.transformer(masked_seq_low)
                    mask_out_seq_low = transformer_dict_low_mask['sequence']
                    transformer_dict_ultralow_mask = self.transformer(masked_seq_ultralow)
                    mask_out_seq_ultralow = transformer_dict_ultralow_mask['sequence']

                    ans_dict['mask_time_window_high_fmri_sequence'] = mask_out_seq_high
                    ans_dict['mask_time_window_low_fmri_sequence'] = mask_out_seq_low
                    ans_dict['mask_time_window_ultralow_fmri_sequence'] = mask_out_seq_ultralow
            
            elif self.masking_method == 'spatial':
                if self.spatial_masking_type == 'hub_ROIs':
                    if self.intermediate_vec == 400:
                        high_comm_list = np.load('./data/communicability/UKB_high_comm_ROI_order_Schaefer400.npy')
                        low_comm_list = np.load('./data/communicability/UKB_low_comm_ROI_order_Schaefer400.npy')
                        ultralow_comm_list = np.load('./data/communicability/UKB_ultralow_comm_ROI_order_Schaefer400.npy')
                    elif self.intermediate_vec == 180:
                        high_comm_list = np.load('./data/communicability/UKB_high_comm_ROI_order_HCP_MMP1.npy')
                        low_comm_list = np.load('./data/communicability/UKB_low_comm_ROI_order_HCP_MMP1.npy')
                        ultralow_comm_list = np.load('./data/communicability/UKB_ultralow_comm_ROI_order_HCP_MMP1.npy')

                    if self.communicability_option == 'remove_high_comm_node':
                        high_mask_list = list(high_comm_list[-self.num_hub_ROIs:])
                        low_mask_list = list(low_comm_list[-self.num_hub_ROIs:])
                        ultralow_mask_list = list(ultralow_comm_list[-self.num_hub_ROIs:])
                    elif self.communicability_option == 'remove_low_comm_node':
                        high_mask_list = list(high_comm_list[:self.num_hub_ROIs])
                        low_mask_list = list(low_comm_list[:self.num_hub_ROIs])
                        ultralow_mask_list = list(ultralow_comm_list[:self.num_hub_ROIs])        
                        
                        
                    for mask in high_mask_list:
                        masked_seq_high[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                    for mask in low_mask_list:
                        masked_seq_low[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                    for mask in ultralow_mask_list:
                        masked_seq_ultralow[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                        
                    transformer_dict_high_mask = self.transformer(masked_seq_high)
                    mask_out_seq_high = transformer_dict_high_mask['sequence']    
                    transformer_dict_low_mask = self.transformer(masked_seq_low)
                    mask_out_seq_low = transformer_dict_low_mask['sequence']
                    transformer_dict_ultralow_mask = self.transformer(masked_seq_ultralow)
                    mask_out_seq_ultralow = transformer_dict_ultralow_mask['sequence']

                    ans_dict['mask_hub_ROIs_high_fmri_sequence'] = mask_out_seq_high
                    ans_dict['mask_hub_ROIs_low_fmri_sequence'] = mask_out_seq_low
                    ans_dict['mask_hub_ROIs_ultralow_fmri_sequence'] = mask_out_seq_ultralow
                    
                elif self.spatial_masking_type == 'random_ROIs':
                    mask_list = random.sample(list(range(self.intermediate_vec)), self.num_random_ROIs)
                    for mask in mask_list:
                        masked_seq_high[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                        masked_seq_low[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                        masked_seq_ultralow[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                    
                    transformer_dict_high_mask = self.transformer(masked_seq_high)
                    mask_out_seq_high = transformer_dict_high_mask['sequence']    
                    transformer_dict_low_mask = self.transformer(masked_seq_low)
                    mask_out_seq_low = transformer_dict_low_mask['sequence']
                    transformer_dict_ultralow_mask = self.transformer(masked_seq_ultralow)
                    mask_out_seq_ultralow = transformer_dict_ultralow_mask['sequence']

                    ans_dict['mask_hub_ROIs_high_fmri_sequence'] = mask_out_seq_high
                    ans_dict['mask_hub_ROIs_low_fmri_sequence'] = mask_out_seq_low
                    ans_dict['mask_hub_ROIs_ultralow_fmri_sequence'] = mask_out_seq_ultralow
                    
            else:
                #spatiotemporal#
                if self.intermediate_vec == 400:
                    high_comm_list = np.load('./data/communicability/UKB_high_comm_ROI_order_Schaefer400.npy')
                    low_comm_list = np.load('./data/communicability/UKB_low_comm_ROI_order_Schaefer400.npy')
                    ultralow_comm_list = np.load('./data/communicability/UKB_ultralow_comm_ROI_order_Schaefer400.npy')
                elif self.intermediate_vec == 180:
                    high_comm_list = np.load('./data/communicability/UKB_high_comm_ROI_order_HCP_MMP1.npy')
                    low_comm_list = np.load('./data/communicability/UKB_low_comm_ROI_order_HCP_MMP1.npy')
                    ultralow_comm_list = np.load('./data/communicability/UKB_ultralow_comm_ROI_order_HCP_MMP1.npy')

                if self.communicability_option == 'remove_high_comm_node':
                    high_mask_list = list(high_comm_list[-self.num_hub_ROIs:])
                    low_mask_list = list(low_comm_list[-self.num_hub_ROIs:])
                    ultralow_mask_list = list(ultralow_comm_list[-self.num_hub_ROIs:])
                elif self.communicability_option == 'remove_low_comm_node':
                    high_mask_list = list(high_comm_list[:self.num_hub_ROIs])
                    low_mask_list = list(low_comm_list[:self.num_hub_ROIs])
                    ultralow_mask_list = list(ultralow_comm_list[:self.num_hub_ROIs])                        
              
                if self.spatiotemporal_masking_type == 'whole':       
                    for mask in high_mask_list:
                        masked_seq_high[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                    for mask in low_mask_list:
                        masked_seq_low[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                    for mask in ultralow_mask_list:
                        masked_seq_ultralow[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence

                    mask_list = list(range(0, self.sequence_length, self.window_interval_rate*self.temporal_masking_window_size))

                    if self.sequence_length - mask_list[-1] < self.temporal_masking_window_size:
                        mask_list = mask_list[:-1]

                    for mask in mask_list:
                        masked_seq_high[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)
                        masked_seq_low[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)
                        masked_seq_ultralow[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)

                    transformer_dict_high_mask = self.transformer(masked_seq_high)
                    mask_out_seq_high = transformer_dict_high_mask['sequence']    
                    transformer_dict_low_mask = self.transformer(masked_seq_low)
                    mask_out_seq_low = transformer_dict_low_mask['sequence']
                    transformer_dict_ultralow_mask = self.transformer(masked_seq_ultralow)
                    mask_out_seq_ultralow = transformer_dict_ultralow_mask['sequence']

                    ans_dict['mask_spatiotemporal_high_fmri_sequence'] = mask_out_seq_high
                    ans_dict['mask_spatiotemporal_low_fmri_sequence'] = mask_out_seq_low
                    ans_dict['mask_spatiotemporal_ultralow_fmri_sequence'] = mask_out_seq_ultralow
                
                elif self.spatiotemporal_masking_type == 'separate':
                    # temporal masking
                    temporal_masked_seq_high = x_h
                    temporal_masked_seq_low = x_l
                    temporal_masked_seq_ultralow = x_u
                    
                    mask_list = list(range(0, self.sequence_length, self.window_interval_rate*self.temporal_masking_window_size))

                    if self.sequence_length - mask_list[-1] < self.temporal_masking_window_size:
                        mask_list = mask_list[:-1]

                    for mask in mask_list:
                        temporal_masked_seq_high[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)
                        temporal_masked_seq_low[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)
                        temporal_masked_seq_ultralow[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(batch_size, self.temporal_masking_window_size, self.intermediate_vec)
                        
                    
                    transformer_dict_high_mask = self.transformer(temporal_masked_seq_high)
                    temporal_mask_out_seq_high = transformer_dict_high_mask['sequence']    
                    transformer_dict_low_mask = self.transformer(temporal_masked_seq_low)
                    temporal_mask_out_seq_low = transformer_dict_low_mask['sequence']
                    transformer_dict_ultralow_mask = self.transformer(temporal_masked_seq_ultralow)
                    temporal_mask_out_seq_ultralow = transformer_dict_ultralow_mask['sequence']

                    ans_dict['temporal_mask_spatiotemporal_high_fmri_sequence'] = temporal_mask_out_seq_high
                    ans_dict['temporal_mask_spatiotemporal_low_fmri_sequence'] = temporal_mask_out_seq_low
                    ans_dict['temporal_mask_spatiotemporal_ultralow_fmri_sequence'] = temporal_mask_out_seq_ultralow

                    
                    # spatial masking
                    
                    spatial_masked_seq_high = x_h
                    spatial_masked_seq_low = x_l
                    spatial_masked_seq_ultralow = x_u
                    
                    for mask in high_mask_list:
                        spatial_masked_seq_high[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                    for mask in low_mask_list:
                        spatial_masked_seq_low[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                    for mask in ultralow_mask_list:
                        spatial_masked_seq_ultralow[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1) # generate masked sequence
                        
                    transformer_dict_high_mask = self.transformer(spatial_masked_seq_high)
                    spatial_mask_out_seq_high = transformer_dict_high_mask['sequence']    
                    transformer_dict_low_mask = self.transformer(spatial_masked_seq_low)
                    spatial_mask_out_seq_low = transformer_dict_low_mask['sequence']
                    transformer_dict_ultralow_mask = self.transformer(spatial_masked_seq_ultralow)
                    spatial_mask_out_seq_ultralow = transformer_dict_ultralow_mask['sequence']

                    ans_dict['spatial_mask_spatiotemporal_high_fmri_sequence'] = spatial_mask_out_seq_high
                    ans_dict['spatial_mask_spatiotemporal_low_fmri_sequence'] = spatial_mask_out_seq_low
                    ans_dict['spatial_mask_spatiotemporal_ultralow_fmri_sequence'] = spatial_mask_out_seq_ultralow
                    
                    
        if self.recon_loss:
            transformer_dict_high = self.transformer(x_h)
            out_seq_high= transformer_dict_high['sequence']
            
            transformer_dict_low = self.transformer(x_l)
            out_seq_low = transformer_dict_low['sequence']
            
            transformer_dict_ultralow = self.transformer(x_u)
            out_seq_ultralow = transformer_dict_ultralow['sequence']
            
            ans_dict['reconstructed_high_fmri_sequence'] = out_seq_high
            ans_dict['reconstructed_low_fmri_sequence'] = out_seq_low
            ans_dict['reconstructed_ultralow_fmri_sequence'] = out_seq_ultralow
            
        return ans_dict