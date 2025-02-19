import torch
import torch.nn as nn
from typing import Dict, Any, Union, List
from collections import defaultdict


class FLOPsCounter:
    """
    PyTorch 모델의 FLOPs를 계산하는 클래스
    """
    def __init__(self):
        self.flops_counter = defaultdict(lambda: 0)
        self.hooks = []
        
    def count_conv2d(self, m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # 입력 텐서
        batch_size = x.size(0)
        output_height = y.size(2)
        output_width = y.size(3)
        
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (x.size(1) // m.groups)
        flops = kernel_ops * m.out_channels * output_height * output_width * batch_size
        
        if m.bias is not None:
            flops += m.out_channels * output_height * output_width * batch_size
            
        self.flops_counter['conv'] += flops
        
    def count_linear(self, m: nn.Linear, x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # 입력 텐서
        batch_size = x.size(0)
        flops = batch_size * m.in_features * m.out_features
        
        if m.bias is not None:
            flops += batch_size * m.out_features
            
        self.flops_counter['linear'] += flops
        
    def count_bn(self, m: Union[nn.BatchNorm1d, nn.BatchNorm2d], x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # 입력 텐서
        flops = x.numel() * 2  # scale과 shift
        self.flops_counter['bn'] += flops
        
    def count_relu(self, m: nn.ReLU, x: torch.Tensor, y: torch.Tensor):
        x = x[0]  # 입력 텐서
        flops = x.numel()
        self.flops_counter['relu'] += flops
        
    def count_attention(self, m: nn.MultiheadAttention, x: torch.Tensor, y: torch.Tensor):
        """Multihead Attention FLOPs 계산 (입력을 정확히 반영한 버전)"""

        # 입력 `x`는 하나의 텐서로 전달됨 → unpack할 필요 없음
        input_tensor = x[0]  # (batch_size, seq_len + 1, hidden_dim)

        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1]  # CLS 토큰 포함해서 seq_len + 1일 수도 있음
        embed_dim = input_tensor.shape[2]
        num_heads = m.num_attention_heads  
        head_dim = embed_dim // num_heads

        # **1. Q, K, V 연산 (입력에서 Query, Key, Value 만들기)**
        flops = 3 * batch_size * seq_len * embed_dim * embed_dim  # W_q, W_k, W_v

        # **2. Q * K^T (Scaled Dot-Product Attention)**
        flops += batch_size * num_heads * seq_len * seq_len * head_dim  # (Q @ K^T)

        # **3. Softmax**
        flops += batch_size * num_heads * seq_len * seq_len  # Softmax 연산

        # **4. Attention * V**
        flops += batch_size * num_heads * seq_len * seq_len * head_dim  # Attention-weighted sum

        # **5. Output Projection (W_o)**
        flops += batch_size * seq_len * embed_dim * embed_dim  # W_o 연산

        self.flops_counter['attention'] += flops

        
    def register_hooks(self, model: nn.Module):
        """모델의 각 레이어에 hook을 등록합니다."""
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
            elif hasattr(module, "num_attention_heads"):  # BERT Attention Layer 감지
                self.hooks.append(module.register_forward_hook(self.count_attention))
        
        model.apply(register_hook)
        
    def remove_hooks(self):
        """등록된 모든 hook을 제거합니다."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def reset_counter(self):
        """FLOPs 카운터를 초기화합니다."""
        self.flops_counter.clear()
        
    def get_total_flops(self) -> int:
        """총 FLOPs를 반환합니다."""
        return sum(self.flops_counter.values())
    
    def print_flops_breakdown(self):
        """레이어 타입별 FLOPs 내역을 출력합니다."""
        print("\nFLOPs breakdown:")
        for layer_type, flops in self.flops_counter.items():
            print(f"{layer_type}: {flops:,} FLOPs ({(flops/self.get_total_flops())*100:.2f}%)")
        print(f"\nTotal FLOPs: {self.get_total_flops():,}")

# 사용 예시
def calculate_model_flops(model, input_size, model_type):
    """
    모델의 FLOPs를 계산합니다.
    
    Args:
        model: PyTorch 모델
        input_size: 입력 텐서의 크기 [batch_size, channels, height, width] 또는
                   [batch_size, sequence_length, hidden_size]
    
    Returns:
        총 FLOPs 수
    """
    flops_counter = FLOPsCounter()
    flops_counter.register_hooks(model)
    
    # 더미 입력으로 모델 실행
    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        if model_type == 'mbbn':
            model(dummy_input, dummy_input, dummy_input)
        elif model_type == 'vanilla_BERT':
            model(dummy_input)
    
    # 결과 출력
    flops_counter.print_flops_breakdown()
    
    # Hook 제거
    flops_counter.remove_hooks()
    
    return flops_counter.get_total_flops()

# 사용 예시:
"""
model = YourModel()
input_size = [1, 3, 224, 224]  # CNN의 경우
# 또는
input_size = [1, 512, 768]  # Transformer의 경우 [batch_size, seq_len, hidden_size]

flops = calculate_model_flops(model, input_size)
"""