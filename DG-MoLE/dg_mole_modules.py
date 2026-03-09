import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AtomExpertLayer(nn.Module):
    def __init__(self, in_features, out_features, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.lora_A = nn.Parameter(torch.empty(num_experts, in_features))
        self.lora_B = nn.Parameter(torch.empty(num_experts, out_features))

        # --- 安全机制 1：初始缩放极小化 ---
        # 即使训练稍微出偏，0.01 的缩放也不会让基座崩溃
        self.scaling = 0.01
        self.reset_parameters()

    def reset_parameters(self):
        # 经典的 LoRA 初始化：A 用高斯，B 用全 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, routing_weights):
        # --- 安全机制 2：强制精度转换与对齐 ---
        # 记录基座的原始类型 (可能是 float16)
        orig_type = x.dtype
        # 统一转为 float32 进行专家计算，避免 int8 溢出
        x_float = x.to(torch.float32)

        # 计算 A 映射
        A_out = torch.einsum('bsi,ei->bse', x_float, self.lora_A.to(torch.float32))

        # 应用路由权重
        weighted_A = A_out * routing_weights.to(torch.float32)

        # 计算 B 映射
        final_output = torch.einsum('bse,eo->bso', weighted_A, self.lora_B.to(torch.float32))

        # --- 安全机制 3：残差保护 ---
        # 返回时转回原始类型，并应用极小的缩放
        return (final_output * self.scaling).to(orig_type)


class SparsegenRouter(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, x):
        # 确保路由器不会因为输入数值大而产生极端分布
        logits = self.gate(x.to(self.gate.weight.dtype))
        return F.softmax(logits, dim=-1)