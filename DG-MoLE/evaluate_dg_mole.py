import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
from model_injector import inject_dg_mole

# --- 1. 全局配置 ---
os.environ['MODEL_SCOPE_HUB_MIRROR'] = 'https://modelscope.cn/api/v1/models'
os.environ["SHOW_ROUTING"] = "0"  # 评测时不打印路由日志，加快速度

MODEL_ID = "qwen/Qwen2.5-7B-Instruct"
SAVE_PATH = "dg_mole_7b_experts_final.pth"

# 尝试导入评估库，如果没有安装则提示
try:
    from rouge_chinese import Rouge
    import jieba

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("⚠️ 未安装 rouge_chinese 或 jieba，Rouge-L 计算将跳过。")
    print("💡 建议运行: pip install rouge_chinese jieba")


def load_eval_system():
    print("=" * 60)
    print("📊 DG-MoLE 自动化评测引擎启动中...")
    print("=" * 60)

    local_path = snapshot_download(MODEL_ID, revision='master')
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = inject_dg_mole(model, num_experts=8, lora_alpha=16.0)

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location="cuda")
        filtered_dict = {k: v for k, v in checkpoint.items() if "router" in k or "experts" in k}
        model.load_state_dict(filtered_dict, strict=False)
        print("✅ 评测权重加载完成！")

    model.eval()
    return tokenizer, model


def calculate_perplexity(model, tokenizer, text):
    """
    计算给定文本的困惑度 (Perplexity, PPL)
    对应论文 3.3 节：衡量模型对特定领域文本的拟合程度
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        ppl = torch.exp(loss).item()
    return ppl


def evaluate_generation(model, tokenizer, test_cases):
    """
    批量生成并计算 Rouge-L 和 PPL
    """
    results = []
    rouge = Rouge() if HAS_ROUGE else None

    for idx, case in enumerate(test_cases):
        instruction = case["instruction"]
        reference = case["reference"]

        print(f"\n[{idx + 1}/{len(test_cases)}] 正在评测任务: {instruction[:20]}...")

        # 1. 构建输入
        messages = [
            {"role": "system", "content": "你是一个集成了 DG-MoLE 架构的 AI 助手。"},
            {"role": "user", "content": instruction}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 2. 生成回答
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # 评测时通常关闭 sample 以保证结果可复现
                temperature=0.0
            )

        response_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        hypothesis = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # 3. 计算 PPL (将参考答案喂给模型算 Loss)
        combined_text = prompt + reference
        ppl = calculate_perplexity(model, tokenizer, combined_text)

        # 4. 计算 Rouge-L
        rouge_l_score = 0.0
        if HAS_ROUGE and hypothesis and reference:
            # 结巴分词处理中文
            hyp_words = ' '.join(jieba.cut(hypothesis))
            ref_words = ' '.join(jieba.cut(reference))
            try:
                scores = rouge.get_scores(hyp_words, ref_words)
                rouge_l_score = scores[0]['rouge-l']['f']
            except ValueError:
                pass  # 防止空字符串报错

        results.append({
            "Instruction": instruction,
            "Hypothesis": hypothesis,
            "Reference": reference,
            "PPL": ppl,
            "Rouge-L": rouge_l_score
        })

    return results


def main():
    tokenizer, model = load_eval_system()

    # --- 模拟多任务测试集 (对应论文 3.2 节) ---
    # 在实际毕设中，这里可以替换为读取 jsonl 测试集文件
    test_cases = [
        {
            "instruction": "法律咨询：租赁合同中逾期利息的上限是多少？",
            "reference": "根据《中华人民共和国民法典》及相关司法解释，租赁合同等民间借贷的逾期利息上限不得超过合同成立时一年期贷款市场报价利率（LPR）的四倍。"
        },
        {
            "instruction": "Python编程：请写一个 BeautifulSoup 抓取 h1 标签的代码。",
            "reference": "使用 requests 获取网页后，使用 BeautifulSoup(html, 'html.parser').find_all('h1') 即可提取所有的 h1 标签。"
        }
    ]

    print("\n⏳ 开始执行自动化评估...")
    results = evaluate_generation(model, tokenizer, test_cases)

    print("\n" + "=" * 50)
    print("📈 DG-MoLE 综合评测报告")
    print("=" * 50)

    avg_ppl = np.mean([r["PPL"] for r in results])
    avg_rouge = np.mean([r["Rouge-L"] for r in results])

    for idx, r in enumerate(results):
        print(f"\n🔹 测试样例 {idx + 1}:")
        print(f"输入: {r['Instruction']}")
        print(f"模型输出: {r['Hypothesis'][:50]}...")
        print(f"样本 PPL: {r['PPL']:.4f}")
        if HAS_ROUGE:
            print(f"样本 Rouge-L: {r['Rouge-L']:.4f}")

    print("-" * 50)
    print(f"🟢 平均困惑度 (PPL): {avg_ppl:.4f}  (越低越好，低于 5.0 说明拟合极佳)")
    if HAS_ROUGE:
        print(f"🔴 平均 Rouge-L F1: {avg_rouge:.4f}  (越高越好，体现文本生成重合度)")
    print("=" * 50)


if __name__ == "__main__":
    main()