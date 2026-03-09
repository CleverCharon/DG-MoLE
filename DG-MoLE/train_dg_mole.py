import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os

# --- 核心：修改为你电脑上的实际路径 ---
model_path = r"C:\Users\Administrator\.cache\modelscope\hub\models\qwen\Qwen2-1___5B-Instruct"

# 1. 配置 8-bit 量化（为了在 6GB 显存上腾出空间跑训练）
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # 允许部分计算溢出到 CPU，防止崩溃
)

print(f"🔄 正在从本地路径读取模型进行量化重载: {model_path}")

# 2. 离线加载量化模型
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,  # 强制不联网
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 3. 注入专家架构
from model_injector import inject_dg_mole

model = inject_dg_mole(model)

# 4. 准备数据
data_samples = [
    {"text": "指导：感冒了应该吃什么药？建议咨询医生并使用对乙酰氨基酚。"},
    {"text": "法律咨询：欠钱不还起诉流程是什么？需要准备起诉状和证据清单。"},
    {"text": "介绍：北京是中国的首都，拥有悠久的历史文化和名胜古迹。"},
    {"text": "医学研究发现，长期熬夜会损害免疫系统并导致记忆力下降。"},
    {"text": "根据劳动法，加班应当支付加班费，且每天不得超过三小时。"}
]


class MoLEDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.encodings = tokenizer([s['text'] for s in samples],
                                   truncation=True, padding='max_length',
                                   max_length=32, return_tensors="pt")

    def __len__(self): return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # 确保数据在 GPU 上
        return {k: v[idx].to("cuda") for k, v in self.encodings.items()}


# 5. 筛选可训练参数并转为 FP32 (量化模型微调必须将 adapter 转为 FP32)
trainable_params = []
for n, p in model.named_parameters():
    if "router" in n or "experts" in n:
        p.requires_grad = True
        trainable_params.append(p)
    else:
        p.requires_grad = False

optimizer = AdamW(trainable_params, lr=1e-3)

# 6. 训练循环
loader = DataLoader(MoLEDataset(data_samples, tokenizer), batch_size=1)
losses = []

print("🚀 8-bit 量化模式已就绪，显存压力已缓解，开始训练...")
model.train()
for epoch in range(10):
    epoch_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        # 计算 Loss
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  # 👈 保护梯度
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/10 | Loss: {avg_loss:.4f}")

# 7. 保存结果
plt.figure()
plt.plot(losses, marker='o')
plt.title("DG-MoLE Training Loss (8-bit Base)")
plt.savefig("loss_curve_final.png")
torch.save(model.state_dict(), "dg_mole_experts_final.pth")
print("✅ 恭喜！训练成功完成，曲线图已生成。")