"""
OSS Personal Assistant — Gradio UI
Powered by Qwen2.5-0.5B-Instruct
Compatible with Gradio 4.x AND 6.x

Run: python app.py
"""
import os
import logging
import gradio as gr

from memory import ConversationMemory
from model import generate, get_model_info
from guardrails import Guardrails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

guardrails = Guardrails(strict=True)
model_info = get_model_info()

SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless AI personal assistant. "
    "Answer questions clearly and concisely. "
    "If you are unsure about a fact, say so — do not make things up. "
    "Politely refuse any requests that are harmful, illegal, or unethical."
)

_gradio_major = int(gr.__version__.split(".")[0])


def create_memory():
    return ConversationMemory(max_turns=10, system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Core chat function — exposed as named API endpoint "/submit"
# ---------------------------------------------------------------------------

def submit(user_message, chat_history, memory, max_tokens, temperature):
    """
    Named '/submit' so gradio_client and the eval runner can call it by name.
    Yields (cleared_input, updated_history, updated_memory).
    """
    if not user_message or not user_message.strip():
        yield "", chat_history, memory
        return

    if memory is None:
        memory = create_memory()

    # Input guardrail
    input_check = guardrails.check_input(user_message)
    if not input_check.allowed:
        bot_reply = guardrails.get_refusal()
        memory.add("user", user_message)
        memory.add("assistant", bot_reply)
        yield "", chat_history + [(user_message, bot_reply)], memory
        return

    memory.add("user", user_message)
    messages = memory.get_messages(include_system=True)

    try:
        response = generate(messages, max_new_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        logger.error("Generation error: %s", e)
        response = "Sorry, I encountered an error. Please try again."

    safe, final_response = guardrails.run(user_message, response)
    if not safe:
        logger.warning("Output blocked by guardrails")

    memory.add("assistant", final_response)
    yield "", chat_history + [(user_message, final_response)], memory


def clear_conversation(memory):
    if memory:
        memory.clear()
    return [], create_memory()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.chatbot { height: 480px; overflow-y: auto; }
footer { display: none !important; }
"""

def _make_chatbot():
    kwargs = {"show_label": False, "elem_classes": ["chatbot"]}
    if _gradio_major < 6:
        kwargs["bubble_full_width"] = False
        kwargs["avatar_images"] = (
            "https://api.dicebear.com/7.x/thumbs/svg?seed=user",
            "https://api.dicebear.com/7.x/thumbs/svg?seed=qwen",
        )
    return gr.Chatbot(**kwargs)


with gr.Blocks(title="OSS Assistant — Qwen2.5") as demo:
    memory_state = gr.State(create_memory)

    gr.Markdown("""
## 🤖 OSS Personal Assistant
**Qwen2.5-0.5B-Instruct · Open Source · Apache 2.0**

Multi-turn assistant with safety guardrails and 10-turn conversation memory.
""")

    chatbot = _make_chatbot()

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask me anything...",
            lines=1, scale=8, show_label=False, container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Accordion("⚙️ Settings", open=False):
        with gr.Row():
            max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
        clear_btn = gr.Button("🗑 Clear conversation", variant="secondary")

    export_out = gr.Textbox(label="Exported conversation", visible=False, lines=10)
    export_btn = gr.Button("📋 Export JSON", variant="secondary")

    with gr.Accordion("ℹ️ Model Info", open=False):
        gr.JSON(model_info)

    # Wire up events — submit function named for gradio_client access
    msg_input.submit(
        submit,
        [msg_input, chatbot, memory_state, max_tokens, temperature],
        [msg_input, chatbot, memory_state],
        api_name="submit",
    )
    send_btn.click(
        submit,
        [msg_input, chatbot, memory_state, max_tokens, temperature],
        [msg_input, chatbot, memory_state],
    )
    clear_btn.click(clear_conversation, [memory_state], [chatbot, memory_state])

    def toggle_export(mem):
        return gr.update(value=mem.export_json() if mem else "{}", visible=True)

    export_btn.click(toggle_export, [memory_state], [export_out])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    launch_kwargs = dict(server_name="0.0.0.0", server_port=port, share=False, show_error=True)
    if _gradio_major >= 6:
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")
        launch_kwargs["css"] = CSS
    else:
        demo.theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")
        demo.css = CSS
    demo.launch(**launch_kwargs)
