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
    def __init__(self,**kwargs):
        super(Spatial_Difference_Loss, self).__init__()
        self.spat_diff_loss = 0.0
        self.spat_diff_loss_type = kwargs.get('spat_diff_loss_type')
        self.fmri_dividing_type = kwargs.get('fmri_dividing_type')
    def forward(self, h, l, u):
        '''
        h, l, u is attention map
        h : (batch, ROI, ROI)
        '''
        loss = nn.L1Loss()
        #loss = nn.MSELoss()
        if self.spat_diff_loss_type == 'minus_log':
            # current SOTA #
            if self.fmri_dividing_type == 'three_channels':
                self.spat_diff_loss = -torch.log((loss(h, l)+loss(h, u)+loss(l, u)))
            elif self.fmri_dividing_type == 'two_channels':
                self.spat_diff_loss = -torch.log(loss(l, u))
        elif self.spat_diff_loss_type == 'reciprocal_log':
            self.spat_diff_loss = torch.tensor(1/(torch.log((loss(h, l)+loss(h, u)+loss(l, u)))))
        elif self.spat_diff_loss_type == 'exp_minus':
            self.spat_diff_loss = torch.tensor((torch.exp(-loss(h, l)) + torch.exp(-loss(h, u)) + torch.exp(-loss(l, u))))
        elif self.spat_diff_loss_type == 'log_loss':
            self.spat_diff_loss = (torch.log(loss(h,l)) + torch.log(loss(h,u)) + torch.log(loss(l,u)))/3
        elif self.spat_diff_loss_type == 'exp_whole':
            self.spat_diff_loss = torch.exp(-1 * (loss(h, l)+loss(h, u)+loss(l, u)))
        self.spat_diff_loss.requires_grad_(True)
        self.spat_diff_loss.retain_grad()
        
        return self.spat_diff_loss
    