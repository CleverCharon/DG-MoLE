import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class DynamicSparseRouter(nn.Module):
    """
    模块一：动态稀疏路由器 (Learnable Dynamic Sparse Routing)
    作用：为每个 Token 计算激活原子专家的权重分布。
    """
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        # 使用不带偏置的线性层作为路由感知机
        self.fc = nn.Linear(hidden_size, num_experts, bias=False)
        # 可学习的稀疏阈值参数 lambda，初始值设为 0.1
        # 对应论文创新点：根据输入文本的复杂度动态调整专家激活数量
        self.sparsity_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 1. 计算原始 logits: [batch_size, seq_len, num_experts]
        logits = self.fc(x)
        
        # 2. Sparsegen 算法的简化可微实现：
        # 基于当前 logits 的均值和动态 lambda 计算出动态截断阈值
        threshold = logits.mean(dim=-1, keepdim=True) * self.sparsity_lambda
        
        # 3. 截断与稀疏化：使用 ReLU 过滤掉低于阈值的专家权重，且保持梯度可导
        routed_weights = F.relu(logits - threshold)
        
        # 4. 归一化：确保被激活的专家权重总和为 1，加上 1e-6 防止分母为零引发 NaN 异常
        routed_weights = routed_weights / (routed_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        return routed_weights


class GranularExperts(nn.Module):
    """
    模块二：细粒度原子专家层 (Granular Expert Construction)
    作用：将传统的 LoRA 低秩矩阵块拆分为多个独立的 Rank-1 向量对。
    """
    def __init__(self, in_features, out_features, num_experts, lora_alpha=16.0):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        # LoRA 的特征缩放因子
        self.scaling = lora_alpha / num_experts 
        
        # 对应论文创新点：将参数分解为独立的 Rank-1 向量对
        # lora_A 负责降维提取特征，lora_B 负责升维输出
        self.lora_A = nn.Parameter(torch.zeros(num_experts, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化策略：A 矩阵采用 Kaiming 均匀分布，B 矩阵全 0 初始化
        # 确保初始状态下旁路对基座模型输出的影响为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, weights):
        """
        前向传播：通过动态路由权重加权组合原子专家
        x 维度: [batch_size, seq_len, in_features]
        weights 维度: [batch_size, seq_len, num_experts]
        """
        # 1. 所有原子专家并行进行输入投影 -> [batch_size, seq_len, num_experts]
        x_proj = F.linear(x, self.lora_A)
        
        # 2. 特征重组：利用路由器算出的权重组合特征 (广播相乘)
        weighted_x = x_proj * weights
        
        # 3. 映射回输出维度 -> [batch_size, seq_len, out_features]
        output = F.linear(weighted_x, self.lora_B.t())
        
        return output * self.scaling


class DGMoLEWrapper(nn.Module):
    """
    模块三：架构包装器
    作用：无缝替换基座模型（如 Qwen/Llama）的 Attention 线性层。
    """
    def __init__(self, original_layer, num_experts=8, lora_alpha=16.0):
        super().__init__()
        self.original_layer = original_layer
        
        # 冻结基座模型的原始参数，保留大模型已有的世界知识
        self.original_layer.requires_grad_(False) 
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 挂载自定义的原子专家库与动态路由器
        self.experts = GranularExperts(in_features, out_features, num_experts, lora_alpha)
        self.router = DynamicSparseRouter(in_features, num_experts)

    def forward(self, x):
        # 1. 计算原始主干网络的输出
        base_output = self.original_layer(x)

        # 2. 数据类型对齐：确保输入 x 的类型与专家网络精度一致 (兼容 BF16/FP16)
        x_float = x.to(self.experts.lora_A.dtype)

        # 3. 路由分配计算
        weights = self.router(x_float)
        
        # 4. 可视化路由监控 (仅在环境变量 SHOW_ROUTING=1 且满足极小概率时打印，防止刷屏)
        if os.getenv("SHOW_ROUTING") == "1" and torch.rand(1).item() < 0.001: 
            # 兼容性修复：转为 float32 后再转 numpy，防止 bfloat16 不受支持报错
            avg_weights = weights.mean(dim=[0, 1]).detach().float().cpu().numpy()
            formatted_w = [f"{w:.3f}" for w in avg_weights]
            print(f"\n[路由监控] 专家激活分布: {formatted_w}")

        # 5. 专家组合输出
        expert_output = self.experts(x_float, weights)

        # 6. 残差连接：将专家特化输出叠加回基础输出，并保持原始的数据类型
        return base_output + expert_output.to(base_output.dtype)


def inject_dg_mole(model, num_experts=8, lora_alpha=16.0):
    """
    模块四：模型结构注入函数
    作用：遍历大语言模型，将 Attention 层的 q_proj 和 v_proj 替换为 DG-MoLE 架构。
    """
    print("🔍 正在初始化 DG-MoLE 架构并扫描目标网络层...")
    injected_count = 0
    
    for name, module in model.named_modules():
        # 针对主流大模型架构，我们默认微调 query (q_proj) 和 value (v_proj) 投影层
        if any(target in name for target in ["q_proj", "v_proj"]):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name)
            
            original_layer = getattr(parent, child_name)
            
            # 防止重复注入
            if isinstance(original_layer, nn.Linear):
                dg_mole_layer = DGMoLEWrapper(original_layer, num_experts, lora_alpha)
                setattr(parent, child_name, dg_mole_layer)
                injected_count += 1
                
    print(f"✅ 架构改造完成！成功向基座模型注入了 {injected_count} 个 DG-MoLE 包装层。")
    return model
