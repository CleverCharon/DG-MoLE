import os
import sys
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from modelscope import snapshot_download

# --- 1. 全局配置与环境变量 ---
os.environ['MODEL_SCOPE_HUB_MIRROR'] = 'https://modelscope.cn/api/v1/models'
# 开启 MoE 路由监控 (对应 model_injector.py 里的判断逻辑)
os.environ["SHOW_ROUTING"] = "1"

MODEL_ID = "qwen/Qwen2.5-7B-Instruct"
SAVE_PATH = "dg_mole_7b_experts_final.pth"


def load_cli_system():
    print("=" * 60)
    print("🚀 DG-MoLE 终端流式推理引擎启动中...")
    print("=" * 60)

    # 1. 下载并加载分词器
    print("📦 正在检查基座模型 (Qwen2.5-7B)...")
    local_path = snapshot_download(MODEL_ID, revision='master')
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    # 2. 加载 BF16 基座模型
    print("🧠 正在装载基座模型到 GPU (BF16 模式)...")
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 3. 注入 DG-MoLE 架构
    from model_injector import inject_dg_mole
    model = inject_dg_mole(model, num_experts=8, lora_alpha=16.0)

    # 4. 精准加载训练好的专家层权重
    if os.path.exists(SAVE_PATH):
        print(f"🛠️ 正在从 {SAVE_PATH} 提取并挂载专家参数...")
        checkpoint = torch.load(SAVE_PATH, map_location="cuda")
        # 过滤逻辑：只加载专家和路由器的参数
        filtered_dict = {k: v for k, v in checkpoint.items() if "router" in k or "experts" in k}
        model.load_state_dict(filtered_dict, strict=False)
        print("✅ DG-MoLE 专家层已成功激活！")
    else:
        print("⚠️ 未发现微调权重文件，将以随机初始化的专家架构运行（仅供调试）。")

    model.eval()
    return tokenizer, model


def main():
    tokenizer, model = load_cli_system()

    print("\n[系统已就绪] 💡 输入您的指令进行测试。")
    print("         输入 'clear' 清空历史，输入 'exit' 退出。")

    history = []

    while True:
        try:
            query = input("\n👤 用户 >>> ").strip()
        except EOFError:
            break

        if not query: continue
        if query.lower() in ['exit', 'quit', '退出']: break
        if query.lower() == 'clear':
            history = []
            print("🧹 对话历史已重置")
            continue

        # --- 1. 严格构建对话历史 (防御性编程) ---
        clean_messages = [{"role": "system", "content": "你是一个集成了 DG-MoLE 专家路由架构的 AI 助手。"}]
        for q, a in history:
            if q and a:  # 确保历史记录不为空
                clean_messages.append({"role": "user", "content": str(q)})
                clean_messages.append({"role": "assistant", "content": str(a)})
        clean_messages.append({"role": "user", "content": str(query)})

        # --- 2. 强制获取字符串文本 (解决 Tokenizer 类型报错) ---
        try:
            text = tokenizer.apply_chat_template(
                clean_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 如果解析出意外的 list 格式，强制解码回字符串
            if isinstance(text, list):
                text = tokenizer.decode(text)
            text = str(text)
        except Exception as e:
            print(f"\n⚠️ 模板解析异常，已降级为基础拼接: {e}")
            text = f"User: {query}\nAssistant:"

        # 转换为 Tensor 输入
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        # --- 3. 设置流式输出器 (TextIteratorStreamer) ---
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

        # --- 4. 在子线程中启动模型生成 (防止阻塞主线程的打印) ---
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("🤖 AI (DG-MoLE) >>> ", end="", flush=True)
        full_response = ""

        # 主线程实时捕获输出并打印 (打字机效果)
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text

        print("\n" + "-" * 40)

        # 更新历史记录，保留最近 5 轮，防止显存爆炸
        history.append((query, full_response))
        if len(history) > 5:
            history.pop(0)


if __name__ == "__main__":
    main()