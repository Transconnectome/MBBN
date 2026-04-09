import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Mask_Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Mask_Loss, self).__init__()
        self.mask_loss = 0.0
        
    def forward(self, input_seq, output_seq):
        '''
        input : raw fmri timeseries. shape is [batch, timelength, 180]
        output : reconstructed sequence. (masked sequence -> transformer -> reconstructed sequence.) shape is [batch, timelength, 180]
        '''

        loss = nn.L1Loss()
        self.mask_loss = loss(input_seq, output_seq)
        self.mask_loss.requires_grad_(True)
        self.mask_loss.retain_grad()
        
        return self.mask_loss
    
    
class Spatial_Difference_Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Spatial_Difference_Loss, self).__init__()

    def forward(self, h, l, u):
        '''
        Penalises similarity between the three band-specific spatial attention maps.
        h, l, u : (batch, ROI, ROI) — high / low / ultralow frequency attention maps
        '''
        loss = nn.L1Loss()
        spat_diff_loss = -torch.log(loss(h, l) + loss(h, u) + loss(l, u))
        spat_diff_loss.requires_grad_(True)
        spat_diff_loss.retain_grad()
        return spat_diff_loss
    