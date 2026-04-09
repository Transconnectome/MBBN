import os
from abc import ABC, abstractmethod
import torch
from transformers import BertConfig, BertPreTrainedModel, BertModel
import torch.nn as nn
import numpy as np


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
        q, k, v = qkv.unbind(0)
        # q, k, v: B, num_heads, N, C // num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn: batch, num_heads, ROI, ROI

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
        self.best_AUROC = 0

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def register_vars(self, **kwargs):
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.spatiotemporal = kwargs.get('spatiotemporal')
        self.transformer_dropout_rate = kwargs.get('transformer_dropout_rate')
        self.sequence_length = kwargs.get('sequence_length')
        self.pretrained_model_weights_path = kwargs.get('pretrained_model_weights_path')
        self.finetune = kwargs.get('finetune')
        self.transfer_learning = bool(self.pretrained_model_weights_path) or self.finetune
        self.finetune_test = kwargs.get('finetune_test')
        self.num_heads = kwargs.get('num_heads')
        self.target = kwargs.get('target')
        self.task = kwargs.get('fine_tune_task')
        self.step = kwargs.get('step')
        self.visualization = kwargs.get('visualization')

        if self.transfer_learning or self.finetune_test:
            self.sequence_length += (464 - self.sequence_length)

        self.BertConfig = BertConfig(
            hidden_size=self.intermediate_vec,
            vocab_size=1,
            num_hidden_layers=kwargs.get('transformer_hidden_layers'),
            num_attention_heads=self.num_heads,
            max_position_embeddings=self.sequence_length + 1,
            hidden_dropout_prob=self.transformer_dropout_rate
        )
        self.label_num = 1
        self.use_cuda = kwargs.get('gpu')
        self.dataset_name = kwargs.get('dataset_name')

    def load_partial_state_dict(self, state_dict, load_cls_embedding):
        print('loading parameters onto new model...')
        own_state = self.state_dict()
        loaded = {name: False for name in own_state.keys()}
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
        for name, was_loaded in loaded.items():
            if not was_loaded:
                print('notice: named parameter - {} is randomly initialized'.format(name))

    def save_checkpoint(self, directory, title, epoch, loss, AUROC, optimizer=None, schedule=None):
        if not os.path.exists(directory):
            os.makedirs(directory)

        ckpt_dict = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'loss_value': loss
        }
        if AUROC is not None:
            ckpt_dict['AUROC'] = AUROC
        if schedule is not None:
            ckpt_dict['schedule_state_dict'] = schedule.state_dict()
            ckpt_dict['lr'] = schedule.get_last_lr()[0]
        if hasattr(self, 'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path

        core_name = title
        name = "{}_last_epoch.pth".format(core_name)
        torch.save(ckpt_dict, os.path.join(directory, name))

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
    def __init__(self, config, **kwargs):
        super(Transformer_Block, self).__init__(config)
        self.register_vars(**kwargs)
        self.cls_pooling = True
        self.init_weights()
        self.bert = BertModel(config, add_pooling_layer=self.cls_pooling)
        self.cls_embedding = nn.Sequential(
            nn.Linear(self.intermediate_vec, self.intermediate_vec), nn.LeakyReLU()
        )
        self.register_buffer('cls_id', (torch.ones((1, 1, self.intermediate_vec)) * 0.5), persistent=False)

    def concatenate_cls(self, x):
        cls_token = self.cls_embedding(self.cls_id.expand(x.size()[0], -1, -1))
        return torch.cat([cls_token, x], dim=1)

    def forward(self, x):
        inputs_embeds = self.concatenate_cls(x)  # (batch, seq_len+1, ROI)
        outputs = self.bert(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True
        )
        sequence_output = outputs[0][:, 1:, :]  # (batch, seq_len, ROI)
        pooled_cls = outputs[1]                  # (batch, ROI)
        return {'sequence': sequence_output, 'cls': pooled_cls}


# Step 1: vanilla BERT baseline (single-band, no frequency decomposition)
class Transformer_Finetune(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Finetune, self).__init__()
        self.register_vars(**kwargs)
        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num)

        if self.spatiotemporal:
            if self.sequence_length % 12 == 0:
                num_heads = 12
            elif self.sequence_length % 8 == 0:
                num_heads = 8
            self.spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
            self.regression_head_spatial = Classifier(
                self.intermediate_vec * self.intermediate_vec, self.label_num
            )

    def forward(self, x):
        # x: (batch, seq_len, ROI)
        transformer_dict = self.transformer(x)

        if self.spatiotemporal:
            spatial_attention = self.spatial_attention(x.permute(0, 2, 1))
            batch_size = spatial_attention.shape[0]
            out_cls_spatial = torch.mean(spatial_attention, dim=1).reshape(batch_size, -1)
            pred_spatial = self.regression_head_spatial(out_cls_spatial)

        out_cls = transformer_dict['cls']
        pred_temporal = self.regression_head(out_cls)

        if self.spatiotemporal:
            prediction = (pred_spatial + pred_temporal) / 2
            ans_dict = {self.task: prediction, 'spatial_attention': spatial_attention}
        else:
            prediction = pred_temporal
            ans_dict = {self.task: prediction}

        return ans_dict


# Step 2: MBBN main model (three frequency bands: high, low, ultralow)
class Transformer_Finetune_Three_Channels(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Finetune_Three_Channels, self).__init__()
        self.register_vars(**kwargs)

        # Shared temporal transformer (parameter sharing across bands)
        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

        # Per-band spatial attention
        if self.sequence_length % 12 == 0:
            num_heads = 12
        elif self.sequence_length % 8 == 0:
            num_heads = 8
        self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)

        self.regression_head = Classifier(self.intermediate_vec, self.label_num)

    def forward(self, x_h, x_l, x_u):
        # Input shape: (batch, seq_len, ROI)

        # Temporal (shared transformer)
        transformer_dict_high = self.transformer(x_h)
        transformer_dict_low = self.transformer(x_l)
        transformer_dict_ultralow = self.transformer(x_u)

        # Spatial
        high_spatial_attention = self.high_spatial_attention(x_h.permute(0, 2, 1))
        low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1))
        ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1))

        pred_high = self.regression_head(transformer_dict_high['cls'])
        pred_low = self.regression_head(transformer_dict_low['cls'])
        pred_ultralow = self.regression_head(transformer_dict_ultralow['cls'])

        prediction = (pred_high + pred_low + pred_ultralow) / 3

        if self.visualization:
            return prediction

        ans_dict = {
            self.task: prediction,
            'high_spatial_attention': high_spatial_attention,
            'low_spatial_attention': low_spatial_attention,
            'ultralow_spatial_attention': ultralow_spatial_attention
        }
        return ans_dict


# Step 3: MBBN pretraining (spatiotemporal masking + reconstruction)
class Transformer_Reconstruction_Three_Channels(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Reconstruction_Three_Channels, self).__init__()

        self.temporal_masking_window_size = kwargs.get('temporal_masking_window_size')
        self.window_interval_rate = kwargs.get('window_interval_rate')
        self.num_hub_ROIs = kwargs.get('num_hub_ROIs')
        self.communicability_option = kwargs.get('communicability_option')
        self.spatiotemporal = kwargs.get('spatiotemporal')

        self.register_vars(**kwargs)

        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

        if self.sequence_length % 12 == 0:
            num_heads = 12
        elif self.sequence_length % 8 == 0:
            num_heads = 8
        self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)
        self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads)

    def forward(self, x_h, x_l, x_u):
        ans_dict = {}

        # Spatial loss
        high_spatial_attention = self.high_spatial_attention(x_h.permute(0, 2, 1))
        low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1))
        ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1))
        ans_dict['high_spatial_attention'] = high_spatial_attention
        ans_dict['low_spatial_attention'] = low_spatial_attention
        ans_dict['ultralow_spatial_attention'] = ultralow_spatial_attention

        # Load communicability-ordered ROI lists
        if self.intermediate_vec == 400:
            high_comm_list = np.load('./data/communicability/UKB_new_high_comm_ROI_order_Schaefer400.npy')
            low_comm_list = np.load('./data/communicability/UKB_new_low_comm_ROI_order_Schaefer400.npy')
            ultralow_comm_list = np.load('./data/communicability/UKB_new_ultralow_comm_ROI_order_Schaefer400.npy')
        elif self.intermediate_vec == 360:
            high_comm_list = np.load('./data/communicability/ABCD_new_high_comm_ROI_order_HCPMMP1_asymmetric.npy')
            low_comm_list = np.load('./data/communicability/ABCD_new_low_comm_ROI_order_HCPMMP1_asymmetric.npy')
            ultralow_comm_list = np.load('./data/communicability/ABCD_new_ultralow_comm_ROI_order_HCPMMP1_asymmetric.npy')

        # Top-k high-communicability hub ROIs to mask
        high_mask_list = list(high_comm_list[-self.num_hub_ROIs:])
        low_mask_list = list(low_comm_list[-self.num_hub_ROIs:])
        ultralow_mask_list = list(ultralow_comm_list[-self.num_hub_ROIs:])

        batch_size = x_h.shape[0]
        device = x_h.device
        masked_seq_high = x_h.clone()
        masked_seq_low = x_l.clone()
        masked_seq_ultralow = x_u.clone()

        # Spatial masking (zero out hub ROI columns)
        for mask in high_mask_list:
            masked_seq_high[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1, device=device)
        for mask in low_mask_list:
            masked_seq_low[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1, device=device)
        for mask in ultralow_mask_list:
            masked_seq_ultralow[:, :, mask:mask+1] = torch.zeros(batch_size, self.sequence_length, 1, device=device)

        # Temporal masking (zero out time windows)
        mask_list = list(range(0, self.sequence_length, self.window_interval_rate * self.temporal_masking_window_size))
        if self.sequence_length - mask_list[-1] < self.temporal_masking_window_size:
            mask_list = mask_list[:-1]
        for mask in mask_list:
            masked_seq_high[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(
                batch_size, self.temporal_masking_window_size, self.intermediate_vec, device=device)
            masked_seq_low[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(
                batch_size, self.temporal_masking_window_size, self.intermediate_vec, device=device)
            masked_seq_ultralow[:, mask:mask+self.temporal_masking_window_size, :] = torch.zeros(
                batch_size, self.temporal_masking_window_size, self.intermediate_vec, device=device)

        transformer_dict_high_mask = self.transformer(masked_seq_high)
        transformer_dict_low_mask = self.transformer(masked_seq_low)
        transformer_dict_ultralow_mask = self.transformer(masked_seq_ultralow)

        ans_dict['mask_spatiotemporal_high_fmri_sequence'] = transformer_dict_high_mask['sequence']
        ans_dict['mask_spatiotemporal_low_fmri_sequence'] = transformer_dict_low_mask['sequence']
        ans_dict['mask_spatiotemporal_ultralow_fmri_sequence'] = transformer_dict_ultralow_mask['sequence']

        return ans_dict
