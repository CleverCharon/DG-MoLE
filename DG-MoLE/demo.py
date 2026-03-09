import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import gc

# 1. 全局变量初始化
tokenizer = None
model = None
MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\qwen\Qwen2-1___5B-Instruct"
SAVE_PATH = "dg_mole_experts_final.pth"


def load_system():
    global tokenizer, model
    print("🔄 系统重载中，正在清空显存缓存...")
    torch.cuda.empty_cache()
    gc.collect()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    from model_injector import inject_dg_mole
    model = inject_dg_mole(model)

    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location="cuda"), strict=False)
        print(f"✅ 权重加载成功！")
    model.eval()


# 2. 预测逻辑
def predict(message):
    input_text = f"User: {message}\nAssistant: "
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # 修改 demo.py 中的 generate 部分
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            repetition_penalty=1.2,  # 👈 关键：加大惩罚，防止胡言乱语
            do_sample=True,  # 👈 开启采样，让回答更自然
            temperature=0.7,  # 👈 降低温度，增加稳定性
            no_repeat_ngram_size=3,  # 👈 防止模型陷入无限循环
            pad_token_id=tokenizer.eos_token_id
        )

    full_res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_res.split("Assistant:")[-1].strip() if "Assistant:" in full_res else full_res


# 3. 核心：手动格式适配逻辑
def chat_handler(message, history):
    bot_message = predict(message)

    # 自动探测你的 Gradio 到底想要什么格式
    # 如果报错提示需要 dictionary，我们就给它字典
    # 如果它依然傲娇，我们就用最原始的元组
    try:
        # 尝试 Gradio 6.0+ 的字典格式
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_message})
    except:
        # 退回到旧版的元组格式
        history.append((message, bot_message))

    return "", history


# 执行加载
load_system()

# 4. 界面构建（使用最基础的组件组合，不使用 ChatInterface 封装）
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎓 DG-MoLE 动态专家系统演示 (兼容模式)")

    # 关键：这里不指定 type，让 Gradio 自己决定
    chatbot = gr.Chatbot(label="对话历史")
    msg = gr.Textbox(label="输入你的问题", placeholder="输入并按回车...")
    clear = gr.Button("🗑️ 清空记录")

    # 绑定提交事件
    msg.submit(chat_handler, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)