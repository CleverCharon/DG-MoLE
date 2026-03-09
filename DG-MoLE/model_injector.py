import torch
from dg_mole_modules import AtomExpertLayer, SparsegenRouter


class DGMoLEWrapper(torch.nn.Module):
    def __init__(self, original_layer, num_experts=8):
        super().__init__()
        self.original_layer = original_layer

        # 获取设备，但数据类型强制指定为 float32 或 float16，不随原层的 int8
        device = original_layer.weight.device
        dtype = torch.float16 if "cuda" in str(device) else torch.float32

        # 专家和路由器必须使用浮点数，不能是 int8
        self.router = SparsegenRouter(original_layer.in_features, num_experts).to(device).to(dtype)
        self.experts = AtomExpertLayer(original_layer.in_features, original_layer.out_features, num_experts).to(
            device).to(dtype)

        # 冻结量化后的原层
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. 原始层计算 (int8 会在内部自动处理)
        base_output = self.original_layer(x)

        # 2. 确保输入 x 的类型与我们的专家层一致 (转为浮点数)
        x_float = x.to(self.experts.lora_A.dtype)

        # 3. 计算路由和专家输出
        weights = self.router(x_float)
        expert_output = self.experts(x_float, weights)

        # 4. 残差求和 (确保类型匹配)
        return base_output + expert_output.to(base_output.dtype)


def inject_dg_mole(model):
    print("开始注入 DG-MoLE 专家层 (量化兼容模式)...")
    # 注意：量化模型的线性层通常是 Linear8bitLt 或类似的类
    for name, module in model.named_modules():
        if "q_proj" in name or "v_proj" in name:
            parent_name = ".".join(name.split(".")[:-1])
            target_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            # 注入包装层
            wrapped_layer = DGMoLEWrapper(module)
            setattr(parent, target_name, wrapped_layer)
    print("注入完成！")
    return model