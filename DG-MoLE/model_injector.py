import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


class DynamicSparseRouter(nn.Module):
    """
    动态稀疏路由器 (对应论文中的 Learnable Dynamic Sparse Routing)
    用于计算每个 Token 对各个原子专家的激活权重。
    """

    def __init__(self, hidden_size, num_experts):
        super().__init__()
        # 使用不带偏置的线性层作为路由感知机
        self.fc = nn.Linear(hidden_size, num_experts, bias=False)
        # 可学习的稀疏阈值参数 lambda (初始值设为 0.1)
        # 对应论文：模型能根据输入文本复杂度动态调整激活专家数量
        self.sparsity_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 计算原始 logits: [batch_size, seq_len, num_experts]
        logits = self.fc(x)

        # Sparsegen 算法的简化可微实现：
        # 根据当前 logits 的均值和动态 lambda 计算出稀疏截断阈值
        threshold = logits.mean(dim=-1, keepdim=True) * self.sparsity_lambda

        # 使用 ReLU 截断低于阈值的值，实现稀疏化且保持梯度可导
        routed_weights = F.relu(logits - threshold)

        # 归一化以确保激活的专家权重和为 1 (加上 epsilon 防止除零 NaN)
        routed_weights = routed_weights / (routed_weights.sum(dim=-1, keepdim=True) + 1e-6)
        return routed_weights


class GranularExperts(nn.Module):
    """
    细粒度原子专家 (对应论文中的 Granular Expert Construction)
    将传统的矩阵块拆分为 N 个 Rank-1 向量对。
    """

    def __init__(self, in_features, out_features, num_experts, lora_alpha=16.0):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        # 对应 LoRA 的缩放因子
        self.scaling = lora_alpha / num_experts

        # 将参数分解为独立的 Rank-1 向量对 (即原子专家)
        # lora_A: [num_experts, in_features]
        self.lora_A = nn.Parameter(torch.zeros(num_experts, in_features))
        # lora_B: [num_experts, out_features]
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化策略：A 采用 Kaiming 均匀分布，B 采用全 0 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, weights):
        """
        前向传播：通过动态权重组合原子专家
        x: [batch_size, seq_len, in_features]
        weights: [batch_size, seq_len, num_experts]
        """
        # 第一步：计算所有专家的输入投影 -> [batch_size, seq_len, num_experts]
        x_proj = F.linear(x, self.lora_A)

        # 第二步：应用路由权重，加权组合激活的专家特征
        weighted_x = x_proj * weights

        # 第三步：映射回输出维度 -> [batch_size, seq_len, out_features]
        output = F.linear(weighted_x, self.lora_B.t())

        return output * self.scaling


class DGMoLEWrapper(nn.Module):
    """
    DG-MoLE 包装器，用于透明替换基座模型(如 Qwen/Llama)的特定 Linear 层
    """

    def __init__(self, original_layer, num_experts=8, lora_alpha=16.0):
        super().__init__()
        self.original_layer = original_layer
        # 冻结基座模型的原始参数，避免破坏通用能力
        self.original_layer.requires_grad_(False)

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 挂载原子专家库与动态路由器
        self.experts = GranularExperts(in_features, out_features, num_experts, lora_alpha)
        self.router = DynamicSparseRouter(in_features, num_experts)

    def forward(self, x):
        # 1. 计算原始层输出 (主干旁路)
        base_output = self.original_layer(x)

        # 2. 统一精度：确保输入 x 的类型与 LoRA 专家精度一致 (通常是 bfloat16)
        x_float = x.to(self.experts.lora_A.dtype)

        # 3. 动态路由器计算出各个专家的权重
        weights = self.router(x_float)

        # --- 4. 可视化路由监控逻辑 (解决之前所有的报错与耦合问题) ---
        # 只有在 cli_demo.py 中开启 SHOW_ROUTING=1 才会打印
        if os.getenv("SHOW_ROUTING") == "1" and torch.rand(1).item() < 0.001:
            # 关键修复：先转为 float() 再给 numpy，解决 BFloat16 兼容性报错
            avg_weights = weights.mean(dim=[0, 1]).detach().float().cpu().numpy()
            formatted_w = [f"{w:.3f}" for w in avg_weights]
            print(f"\n[路由监控] 专家激活分布: {formatted_w}")
        # -------------------------------------------------------------

        # 5. 计算混合专家特征输出
        expert_output = self.experts(x_float, weights)

        # 6. 残差连接求和并返回
        return base_output + expert_output.to(base_output.dtype)


def inject_dg_mole(model, num_experts=8, lora_alpha=16.0):
    """
    模型注入函数：遍历大语言模型，将 Attention 层的 q_proj 和 v_proj 替换为 DG-MoLE 架构
    """
    print("🔍 正在初始化 DG-MoLE 架构并扫描目标网络层...")
    injected_count = 0
    for name, module in model.named_modules():
        # 针对 Transformer 架构，默认替换 query 和 value 投影层
        if any(target in name for target in ["q_proj", "v_proj"]):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name)

            original_layer = getattr(parent, child_name)

            # 确保不重复替换
            if isinstance(original_layer, nn.Linear):
                dg_mole_layer = DGMoLEWrapper(original_layer, num_experts, lora_alpha)
                setattr(parent, child_name, dg_mole_layer)
                injected_count += 1

    print(f"✅ 架构改造完成！成功向基座模型注入了 {injected_count} 个 DG-MoLE 包装层。")
    return model