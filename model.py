import os
from abc import ABC, abstractmethod
import torch
from transformers import BertConfig, BertPreTrainedModel, BertModel
import torch.nn as nn
import numpy as np


def pick_spatial_heads(sequence_length, preferred=(12, 8)):
    """Choose a head count that divides the spatial-attention dimension.

    The released code tried 12 then 8 with no fallback, so any sequence length
    divisible by neither raised UnboundLocalError at construction time.
    """
    for h in preferred:
        if sequence_length % h == 0:
            return h
    for h in range(min(preferred), 0, -1):
        if sequence_length % h == 0:
            return h
    return 1


class Attention(nn.Module):
    '''
    N = ROIs, C = sequence length

    Returns the ROI x ROI attention matrix (`return_attn=True`, the default and
    what MBBN uses as its band-specific connectivity estimate). The feature path
    is also functional now: the released version referenced `self.proj` /
    `self.proj_drop`, which were never created, so `return_attn=False` raised
    AttributeError.

    `attn_only=True` skips the value projection entirely. When only the
    attention matrix is consumed, `v` is computed and discarded, which is a
    third of this module's parameters and FLOPs.
    '''
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 attn_only=False):
        super().__init__()
        self.num_heads = num_heads
        self.attn_only = attn_only
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * (2 if attn_only else 3), bias=qkv_bias)
        self.drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        if not attn_only:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

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
        n_parts = 2 if self.attn_only else 3
        qkv = self.qkv(x).reshape(B, N, n_parts, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if self.attn_only:
            q, k = qkv.unbind(0)
            v = None
        else:
            q, k, v = qkv.unbind(0)
        # q, k, v: B, num_heads, N, C // num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn: batch, num_heads, ROI, ROI

        if return_attn:
            return attn
        if v is None:
            raise RuntimeError('Attention was built with attn_only=True; it has no '
                               'value projection and cannot return features.')
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Classifier(nn.Module):
    """Prediction head.

    `head_type='published'` reproduces the released head exactly:
    ``Linear -> BatchNorm1d(out_features) -> Dropout(p)`` applied to the *final
    logit*. With ``out_features=1`` that has two measured consequences (see
    ``docs/audit/AUDIT.md``): batch-norming a scalar makes one subject's logit
    depend on who else is in the batch (SD 0.29 at batch 16, sigmoid p spanning
    0.107-0.464 for a fixed subject and fixed weights), and ``Dropout(0.6)``
    sets 61% of training logits to exactly 0, i.e. p=0.5 regardless of input.

    `head_type='linear'` (default) is a plain linear read-out.
    `head_type='mlp'` adds one hidden layer with normalisation applied to the
    *features* rather than the logit, which is where normalisation belongs.
    """

    def __init__(self, in_features, out_features, dropout=0.6, head_type='linear',
                 hidden=None):
        super(Classifier, self).__init__()
        self.head_type = head_type
        if head_type == 'published':
            self.linear = nn.Linear(in_features, out_features)
            self.norm = nn.BatchNorm1d(out_features)
            self.dropout = nn.Dropout(dropout)
        elif head_type == 'linear':
            self.linear = nn.Linear(in_features, out_features)
        elif head_type == 'mlp':
            hidden = hidden or max(out_features, min(256, in_features // 2))
            self.pre = nn.Sequential(nn.LayerNorm(in_features),
                                     nn.Linear(in_features, hidden),
                                     nn.GELU(),
                                     nn.Dropout(dropout))
            self.linear = nn.Linear(hidden, out_features)
        else:
            raise ValueError(f'unknown head_type {head_type!r}')

    def forward(self, x):
        if self.head_type == 'published':
            return self.dropout(self.norm(self.linear(x)))
        if self.head_type == 'mlp':
            return self.linear(self.pre(x))
        return self.linear(x)


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
        self.head_type = kwargs.get('head_type', 'linear')
        self.head_dropout = kwargs.get('head_dropout', 0.6)
        self.band_embedding = bool(kwargs.get('band_embedding', False))
        self.spatial_head = bool(kwargs.get('spatial_head', False))
        self.attn_only = bool(kwargs.get('attn_only', False))
        self.use_padding_mask = bool(kwargs.get('use_padding_mask', True))

        # The released line `self.sequence_length += (464 - self.sequence_length)`
        # is an assignment to the hard-coded UKB length. Whether that matches the
        # checkpoint depends on what the pretraining run actually used, and
        # `pretrain_MBBN.slurm` passes `--sequence_length_phase4`, which
        # sort_args() drops for step 3 -- so pretraining ran at 348, not 464, and
        # loading it into a 464-wide fine-tuning model raises RuntimeError.
        # Make the target length explicit instead of assumed.
        if self.transfer_learning or self.finetune_test:
            self.pretrained_sequence_length = kwargs.get('pretrained_sequence_length') or 464
            self.sequence_length = int(self.pretrained_sequence_length)
        else:
            self.pretrained_sequence_length = None

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
        self.bert = BertModel(config, add_pooling_layer=self.cls_pooling)
        self.cls_embedding = nn.Sequential(
            nn.Linear(self.intermediate_vec, self.intermediate_vec), nn.LeakyReLU()
        )
        # Two defects here. (1) The released code called init_weights() *before*
        # self.bert and self.cls_embedding existed, so it walked an empty module
        # tree and initialised nothing. (2) init_weights() is not the documented
        # entry point: post_init() also builds the tied-weights mapping that
        # transformers >= 5 requires, and without it construction raises
        # AttributeError: 'Transformer_Block' object has no attribute
        # 'all_tied_weights_keys'. So the released model does not even
        # instantiate on a current transformers release.
        if hasattr(self, 'post_init'):
            self.post_init()
        else:                                          # transformers < 4.6
            self.init_weights()
        self.register_buffer('cls_id', (torch.ones((1, 1, self.intermediate_vec)) * 0.5), persistent=False)

    def concatenate_cls(self, x):
        cls_token = self.cls_embedding(self.cls_id.expand(x.size()[0], -1, -1))
        return torch.cat([cls_token, x], dim=1)

    def forward(self, x, valid_mask=None, band_offset=None):
        if band_offset is not None:
            # additive band identity, so a shared encoder can tell the three
            # filtered views apart (segment-embedding style)
            x = x + band_offset.view(1, 1, -1)
        inputs_embeds = self.concatenate_cls(x)  # (batch, seq_len+1, ROI)
        attention_mask = None
        if valid_mask is not None and self.use_padding_mask:
            # ABIDE/ABCD are zero-padded up to the pretraining length; with
            # attention_mask=None the encoder and the CLS pooling attend over
            # those pure-zero timepoints as if they were data.
            vm = valid_mask.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            cls_col = torch.ones(vm.shape[0], 1, dtype=vm.dtype, device=vm.device)
            attention_mask = torch.cat([cls_col, vm], dim=1)
        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
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
        # same head class as MBBN, so a step-1 vs step-2 comparison isolates the
        # frequency decomposition instead of also swapping Linear for
        # Linear+BatchNorm+Dropout
        self.regression_head = Classifier(self.intermediate_vec, self.label_num,
                                          dropout=self.head_dropout, head_type=self.head_type)

        if self.spatiotemporal:
            num_heads = pick_spatial_heads(self.sequence_length)
            self.spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads,
                                               attn_only=self.attn_only)
            self.regression_head_spatial = Classifier(
                self.intermediate_vec * self.intermediate_vec, self.label_num,
                dropout=self.head_dropout, head_type=self.head_type
            )

    def forward(self, x, valid_mask=None):
        # x: (batch, seq_len, ROI)
        transformer_dict = self.transformer(x, valid_mask=valid_mask)

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
        num_heads = pick_spatial_heads(self.sequence_length)
        self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads,
                                                attn_only=self.attn_only)
        self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads,
                                               attn_only=self.attn_only)
        self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads,
                                                    attn_only=self.attn_only)

        self.regression_head = Classifier(self.intermediate_vec, self.label_num,
                                          dropout=self.head_dropout, head_type=self.head_type)

        # The temporal encoder and the read-out head are shared across bands and
        # nothing else marks which band a representation came from, so the
        # temporal stream is a 3x ensemble of one identical function. A learned
        # additive offset per band costs 3*ROI parameters and removes that.
        if self.band_embedding:
            self.band_embed = nn.Parameter(torch.zeros(3, self.intermediate_vec))

        # In the released model the spatial-attention maps feed only the
        # band-repulsion loss: d(prediction)/d(spatial qkv) is exactly 0, so the
        # maps that get interpreted as disorder signatures carry no label
        # gradient. Enable this to put them on the prediction path, which is a
        # precondition for any label-based attribution over them.
        if self.spatial_head:
            self.regression_head_spatial = Classifier(
                self.intermediate_vec * self.intermediate_vec, self.label_num,
                dropout=self.head_dropout, head_type=self.head_type)

    def _band_offset(self, i):
        return self.band_embed[i] if self.band_embedding else None

    def forward(self, x_h, x_l, x_u, valid_mask=None):
        # Input shape: (batch, seq_len, ROI)

        # Temporal (shared transformer)
        transformer_dict_high = self.transformer(x_h, valid_mask=valid_mask,
                                                 band_offset=self._band_offset(0))
        transformer_dict_low = self.transformer(x_l, valid_mask=valid_mask,
                                                band_offset=self._band_offset(1))
        transformer_dict_ultralow = self.transformer(x_u, valid_mask=valid_mask,
                                                     band_offset=self._band_offset(2))

        # Spatial
        high_spatial_attention = self.high_spatial_attention(x_h.permute(0, 2, 1))
        low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1))
        ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1))

        pred_high = self.regression_head(transformer_dict_high['cls'])
        pred_low = self.regression_head(transformer_dict_low['cls'])
        pred_ultralow = self.regression_head(transformer_dict_ultralow['cls'])

        prediction = (pred_high + pred_low + pred_ultralow) / 3

        if self.spatial_head:
            B = x_h.shape[0]
            pred_spatial = sum(
                self.regression_head_spatial(a.mean(dim=1).reshape(B, -1))
                for a in (high_spatial_attention, low_spatial_attention,
                          ultralow_spatial_attention)) / 3
            prediction = (prediction + pred_spatial) / 2

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

        num_heads = pick_spatial_heads(self.sequence_length)
        self.high_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads,
                                                attn_only=self.attn_only)
        self.low_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads,
                                               attn_only=self.attn_only)
        self.ultralow_spatial_attention = Attention(dim=self.sequence_length, num_heads=num_heads,
                                                    attn_only=self.attn_only)

        self.random_mask = bool(kwargs.get('random_mask', False))
        self.mask_ratio_spatial = kwargs.get('mask_ratio_spatial')
        self.mask_ratio_temporal = kwargs.get('mask_ratio_temporal')
        self.comm_dir = kwargs.get('communicability_dir') or './data/communicability'
        self.comm_dataset = kwargs.get('communicability_dataset') or 'UKB'

        # A learned fill value per ROI. Zeroing masked positions is ambiguous:
        # the inputs are z-scored, so 0 is also a perfectly ordinary value and
        # the encoder cannot tell "hidden" from "at the mean".
        self.mask_token = nn.Parameter(torch.zeros(self.intermediate_vec))

        # The communicability order was np.load'ed inside forward(), i.e. three
        # reads from disk per batch, with a path relative to the cwd.
        self.register_buffer('hub_order', self._load_hub_order(), persistent=False)

    def _load_hub_order(self):
        names = [n for n in ({400: 'Schaefer400', 360: 'HCPMMP1_asymmetric',
                              180: 'HCPMMP1'}.get(self.intermediate_vec),
                             f'ROI{self.intermediate_vec}') if n]
        orders, tried = [], []
        for band in ('high', 'low', 'ultralow'):
            found = None
            for atlas in names:
                p = os.path.join(self.comm_dir,
                                 f'{self.comm_dataset}_new_{band}_comm_ROI_order_{atlas}.npy')
                tried.append(p)
                if os.path.exists(p):
                    found = p
                    break
            if found is None:
                raise FileNotFoundError(
                    'no communicability ROI order found. Generate it with '
                    f'`python communicability.py --dataset_name {self.comm_dataset} '
                    f'--ROI_num {self.intermediate_vec}`. Looked for: {tried}')
            order = np.load(found)
            if order.shape[-1] != self.intermediate_vec:
                raise ValueError(f'{found} has {order.shape[-1]} entries but '
                                 f'intermediate_vec={self.intermediate_vec}')
            orders.append(order)
        return torch.as_tensor(np.stack(orders), dtype=torch.long)

    def _spatial_mask(self, batch_size, device, band_idx):
        """(batch, ROI) bool, True where the ROI column is hidden."""
        n_roi = self.intermediate_vec
        k = self.num_hub_ROIs if self.mask_ratio_spatial is None else \
            int(round(float(self.mask_ratio_spatial) * n_roi))
        k = max(0, min(k, n_roi))
        m = torch.zeros(batch_size, n_roi, dtype=torch.bool, device=device)
        if k == 0:
            return m
        if self.random_mask:
            # independent draw per subject, so the model cannot memorise which
            # ROIs are hidden
            idx = torch.rand(batch_size, n_roi, device=device).argsort(dim=1)[:, :k]
            m.scatter_(1, idx, True)
        else:
            # published behaviour: the same top-k communicability hubs for every
            # subject and every epoch
            hubs = self.hub_order[band_idx].to(device)[-k:]
            m[:, hubs] = True
        return m

    def _temporal_mask(self, batch_size, device):
        """(batch, time) bool, True where the timepoint is hidden."""
        T, W = self.sequence_length, self.temporal_masking_window_size
        m = torch.zeros(batch_size, T, dtype=torch.bool, device=device)
        if self.random_mask:
            ratio = 1.0 / self.window_interval_rate if self.mask_ratio_temporal is None \
                else float(self.mask_ratio_temporal)
            n_win = max(1, int(round(ratio * T / W)))
            starts = torch.randint(0, max(1, T - W + 1), (batch_size, n_win), device=device)
            offs = torch.arange(W, device=device)
            idx = (starts.unsqueeze(-1) + offs).clamp_(max=T - 1).reshape(batch_size, -1)
            m.scatter_(1, idx, True)
        else:
            # published behaviour: windows always at 0, r*W, 2*r*W, ...
            starts = list(range(0, T, self.window_interval_rate * W))
            if starts and T - starts[-1] < W:
                starts = starts[:-1]
            for s in starts:
                m[:, s:s + W] = True
        return m

    def _apply_mask(self, x, band_idx):
        B, T, R = x.shape
        sm = self._spatial_mask(B, x.device, band_idx).unsqueeze(1).expand(B, T, R)
        tm = self._temporal_mask(B, x.device).unsqueeze(-1).expand(B, T, R)
        mask = sm | tm
        fill = self.mask_token.view(1, 1, R).expand(B, T, R).to(x.dtype)
        return torch.where(mask, fill, x), mask

    def forward(self, x_h, x_l, x_u, valid_mask=None):
        ans_dict = {}

        # Spatial loss
        high_spatial_attention = self.high_spatial_attention(x_h.permute(0, 2, 1))
        low_spatial_attention = self.low_spatial_attention(x_l.permute(0, 2, 1))
        ultralow_spatial_attention = self.ultralow_spatial_attention(x_u.permute(0, 2, 1))
        ans_dict['high_spatial_attention'] = high_spatial_attention
        ans_dict['low_spatial_attention'] = low_spatial_attention
        ans_dict['ultralow_spatial_attention'] = ultralow_spatial_attention

        masked_seq_high, mask_high = self._apply_mask(x_h, 0)
        masked_seq_low, mask_low = self._apply_mask(x_l, 1)
        masked_seq_ultralow, mask_ultralow = self._apply_mask(x_u, 2)

        transformer_dict_high_mask = self.transformer(masked_seq_high, valid_mask=valid_mask)
        transformer_dict_low_mask = self.transformer(masked_seq_low, valid_mask=valid_mask)
        transformer_dict_ultralow_mask = self.transformer(masked_seq_ultralow, valid_mask=valid_mask)

        ans_dict['mask_spatiotemporal_high_fmri_sequence'] = transformer_dict_high_mask['sequence']
        ans_dict['mask_spatiotemporal_low_fmri_sequence'] = transformer_dict_low_mask['sequence']
        ans_dict['mask_spatiotemporal_ultralow_fmri_sequence'] = transformer_dict_ultralow_mask['sequence']

        # hand the masks to the loss so reconstruction can be scored on hidden
        # positions only (see Mask_Loss.masked_only)
        ans_dict['mask_high'] = mask_high
        ans_dict['mask_low'] = mask_low
        ans_dict['mask_ultralow'] = mask_ultralow

        return ans_dict
