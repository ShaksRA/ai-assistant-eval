"""
Frontier Personal Assistant — Gradio UI
Powered by Claude Sonnet (Anthropic)
Compatible with Gradio 4.x AND 6.x

Run: 
  $env:ANTHROPIC_API_KEY="sk-ant-..."   (PowerShell)
  python app.py
"""
import os
import logging
import gradio as gr

from memory import ConversationMemory
from model import generate, generate_streaming, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless AI personal assistant. "
    "Answer questions clearly and concisely. "
    "Use tools when appropriate — especially for math and time queries. "
    "If you are unsure about a fact, say so. "
    "Politely refuse any requests that are harmful, illegal, or unethical."
)

model_info = get_model_info()
_gradio_major = int(gr.__version__.split(".")[0])


def create_memory():
    return ConversationMemory(max_turns=10, system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Core chat function — exposed as named API endpoint "/submit"
# ---------------------------------------------------------------------------

def submit(user_message, chat_history, memory, max_tokens, temperature, use_tools):
    """
    Named '/submit' so gradio_client and the eval runner can call it by name.
    Yields (cleared_input, updated_history, updated_memory).
    """
    if not user_message or not user_message.strip():
        yield "", chat_history, memory
        return

    if memory is None:
        memory = create_memory()

    memory.add("user", user_message)
    messages = memory.get_messages(include_system=True)

    try:
        if use_tools:
            response = generate(
                messages, max_tokens=max_tokens, temperature=temperature,
                use_tools=True, system_prompt=SYSTEM_PROMPT,
            )
            memory.add("assistant", response)
            yield "", chat_history + [(user_message, response)], memory
        else:
            partial = ""
            new_history = chat_history + [(user_message, "▌")]
            for chunk in generate_streaming(messages, max_tokens=max_tokens):
                partial += chunk
                new_history[-1] = (user_message, partial + "▌")
                yield "", new_history, memory
            new_history[-1] = (user_message, partial)
            memory.add("assistant", partial)
            yield "", new_history, memory

    except Exception as e:
        logger.error("Generation error: %s", e)
        msg = f"⚠️ Error: {e}"
        if "ANTHROPIC_API_KEY" in str(e) or "api_key" in str(e).lower():
            msg = "⚠️ API key not set. Run: $env:ANTHROPIC_API_KEY='sk-ant-...' then restart."
        memory.add("assistant", msg)
        yield "", chat_history + [(user_message, msg)], memory


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
            "https://api.dicebear.com/7.x/thumbs/svg?seed=user2",
            "https://api.dicebear.com/7.x/thumbs/svg?seed=claude",
        )
    return gr.Chatbot(**kwargs)


with gr.Blocks(title="Frontier Assistant — Claude Sonnet") as demo:
    memory_state = gr.State(create_memory)

    gr.Markdown("""
## ✨ Frontier Personal Assistant
**Claude Sonnet · Anthropic · Tool-augmented**

Multi-turn assistant with tool use (calculator · datetime · unit converter) and 10-turn memory.
""")

    chatbot = _make_chatbot()

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask me anything — math, facts, tasks, analysis...",
            lines=1, scale=8, show_label=False, container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Accordion("⚙️ Settings", open=False):
        with gr.Row():
            max_tokens = gr.Slider(64, 2048, value=1024, step=64, label="Max tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
        use_tools = gr.Checkbox(value=True, label="Enable tool use (calculator, datetime, converter)")
        clear_btn = gr.Button("🗑 Clear conversation", variant="secondary")

    export_out = gr.Textbox(label="Exported conversation", visible=False, lines=10)
    export_btn = gr.Button("📋 Export JSON", variant="secondary")

    with gr.Accordion("ℹ️ Model Info", open=False):
        gr.JSON(model_info)

    gr.Examples(
        examples=[
            ["What is the square root of 1764?"],
            ["What time is it in Mumbai right now?"],
            ["Convert 100 km/h to mph"],
            ["Explain the difference between RAM and ROM in simple terms"],
            ["Write a Python function to check if a number is prime"],
        ],
        inputs=msg_input,
        label="Try these examples",
    )

    # Wire events — submit named for gradio_client access
    msg_input.submit(
        submit,
        [msg_input, chatbot, memory_state, max_tokens, temperature, use_tools],
        [msg_input, chatbot, memory_state],
        api_name="submit",
    )
    send_btn.click(
        submit,
        [msg_input, chatbot, memory_state, max_tokens, temperature, use_tools],
        [msg_input, chatbot, memory_state],
    )
    clear_btn.click(clear_conversation, [memory_state], [chatbot, memory_state])

    def toggle_export(mem):
        return gr.update(value=mem.export_json() if mem else "{}", visible=True)

    export_btn.click(toggle_export, [memory_state], [export_out])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    launch_kwargs = dict(server_name="0.0.0.0", server_port=port, share=False, show_error=True)
    if _gradio_major >= 6:
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone")
        launch_kwargs["css"] = CSS
    else:
        demo.theme = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone")
        demo.css = CSS
    demo.launch(**launch_kwargs)
