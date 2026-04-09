import torch
import torch.nn as nn
from typing import Dict, Any, Union, List
from collections import defaultdict


class FLOPsCounter:
    def __init__(self):
        self.flops_counter = defaultdict(lambda: 0)
        self.hooks = []
        
    def count_conv2d(self, m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # input tensor
        batch_size = x.size(0)
        output_height = y.size(2)
        output_width = y.size(3)
        
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (x.size(1) // m.groups)
        flops = kernel_ops * m.out_channels * output_height * output_width * batch_size
        
        if m.bias is not None:
            flops += m.out_channels * output_height * output_width * batch_size
            
        self.flops_counter['conv'] += flops
        
    def count_linear(self, m: nn.Linear, x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # input tensor
        batch_size = x.size(0)
        flops = batch_size * m.in_features * m.out_features
        
        if m.bias is not None:
            flops += batch_size * m.out_features
            
        self.flops_counter['linear'] += flops
        
    def count_bn(self, m: Union[nn.BatchNorm1d, nn.BatchNorm2d], x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # input tensor
        flops = x.numel() * 2  # scale & shift
        self.flops_counter['bn'] += flops
        
    def count_relu(self, m: nn.ReLU, x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # input tensor
        flops = x.numel()
        self.flops_counter['relu'] += flops
        
    def count_attention(self, m: nn.MultiheadAttention, x: torch.Tensor, y: torch.Tensor):
        """Multihead Attention FLOPs"""
        input_tensor = x[0]  # (batch_size, seq_len + 1, hidden_dim)

        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1]  # [CLS] -> then seq_len + 1
        embed_dim = input_tensor.shape[2]
        num_heads = m.num_attention_heads  
        head_dim = embed_dim // num_heads

        # **1. Q, K, V (Input -> Query, Key, Value)**
        flops = 3 * batch_size * seq_len * embed_dim * embed_dim  # W_q, W_k, W_v

        # **2. Q * K^T (Scaled Dot-Product Attention)**
        flops += batch_size * num_heads * seq_len * seq_len * head_dim  # (Q @ K^T)

        # **3. Softmax**
        flops += batch_size * num_heads * seq_len * seq_len 

        # **4. Attention * V**
        flops += batch_size * num_heads * seq_len * seq_len * head_dim  # Attention-weighted sum

        # **5. Output Projection (W_o)**
        flops += batch_size * seq_len * embed_dim * embed_dim 

        self.flops_counter['attention'] += flops

        
    def register_hooks(self, model: nn.Module):
        """register hook for each layers"""
        def register_hook(module: nn.Module):
            if isinstance(module, nn.Conv2d):
                self.hooks.append(
                    module.register_forward_hook(self.count_conv2d)
                )
            elif isinstance(module, nn.Linear):
                self.hooks.append(
                    module.register_forward_hook(self.count_linear)
                )
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.hooks.append(
                    module.register_forward_hook(self.count_bn)
                )
            elif isinstance(module, nn.ReLU):
                self.hooks.append(
                    module.register_forward_hook(self.count_relu)
                )
            elif hasattr(module, "num_attention_heads"):  # BERT Attention Layer!!
                self.hooks.append(module.register_forward_hook(self.count_attention))
        
        model.apply(register_hook)
        
    def remove_hooks(self):
        """removed registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def reset_counter(self):
        """initialize FLOPs counter"""
        self.flops_counter.clear()
        
    def get_total_flops(self) -> int:
        """return total FLOPs"""
        return sum(self.flops_counter.values())
    
    def print_flops_breakdown(self):
        """FLOPs per layer"""
        print("\nFLOPs breakdown:")
        for layer_type, flops in self.flops_counter.items():
            print(f"{layer_type}: {flops:,} FLOPs ({(flops/self.get_total_flops())*100:.2f}%)")
        print(f"\nTotal FLOPs: {self.get_total_flops():,}")

def calculate_model_flops(model, input_size, model_type):
    flops_counter = FLOPsCounter()
    flops_counter.register_hooks(model)
    
    # run model with dummy input
    dummy_input = torch.randn(*input_size).to('cuda')
    with torch.no_grad():
        if model_type == 'mbbn':
            model(dummy_input, dummy_input, dummy_input)
        elif model_type == 'vanilla_BERT':
            model(dummy_input)
    
    # print results
    flops_counter.print_flops_breakdown()
    
    # remove hook
    flops_counter.remove_hooks()
    
    return flops_counter.get_total_flops()
