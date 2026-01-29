import gradio as gr

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None


MODEL_ID = "optimum-intel-internal-testing/tiny-random-t5"


def load_model():
    """
    为了避免在示例环境中长时间下载模型，本函数在依赖缺失时仅返回占位提示。
    在真实部署中，可取消注释模型加载代码以完成实际推理。
    """
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        return None, None

    # 真实使用时可启用以下代码：
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    # return tokenizer, model

    return None, None


tokenizer, model = load_model()


def t5_infer(input_text: str, max_length: int = 64):
    """
    WebUI 回调函数。
    这里刻意不执行真实推理，而是返回一个结构化的占位结果，
    以便前端可视化展示完整流程。
    """
    if not input_text.strip():
        return "请输入需要进行序列到序列转换的文本。", "—"

    if tokenizer is None or model is None:
        pseudo_output = f"[占位结果] 已接收到输入文本：{input_text[:80]}..."
        detail = (
            "当前示例环境未实际下载与加载模型，仅用于展示 WebUI 的交互逻辑与"
            "可视化流程。在线部署时，可在本函数中启用 transformers 的推理代码。"
        )
        return pseudo_output, detail

    # 示例推理逻辑（默认关闭）：
    # inputs = tokenizer(input_text, return_tensors=\"pt\")
    # outputs = model.generate(**inputs, max_length=max_length)
    # text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return text, \"真实推理结果\"

    return "推理功能已关闭（示例环境）。", "请在生产环境中启用真实推理代码。"


def build_demo():
    with gr.Blocks(title="tiny-random-t5 WebUI") as demo:
        gr.Markdown(
            """# tiny-random-t5 WebUI 原型界面

本界面主要用于教学与培训情境下演示序列到序列模型的端到端推理流程，
读者可以在课堂或企业内训中借助该界面说明编码器—解码器结构与参数设置对结果分布的影响。
"""
        )

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="教学示例输入文本",
                    placeholder="例如：请学生给出一个英文句子，并在课堂上现场演示其翻译或改写过程……",
                    lines=5,
                )
                max_len = gr.Slider(
                    8,
                    256,
                    value=64,
                    step=4,
                    label="生成最大长度",
                )
                run_btn = gr.Button("课堂演示（示例）", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="示例输出（占位）",
                    lines=5,
                    interactive=False,
                )
                explain_text = gr.Textbox(
                    label="教学说明",
                    lines=6,
                    interactive=False,
                )

        run_btn.click(
            fn=t5_infer,
            inputs=[input_text, max_len],
            outputs=[output_text, explain_text],
        )

        gr.Markdown(
            """## 说明

- 本教学 Demo 仅演示 WebUI 与推理后端之间的交互关系，不附带真实模型权重。
- 教师可在此基础上引导学生阅读主仓库 tiny-random-t5-WebUI 的详细文档，理解完整工程结构。
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
