"""
OSS model inference — Qwen2.5-0.5B-Instruct via HuggingFace Transformers.

Design decisions:
- Uses the `transformers` pipeline API for simplicity
- Supports both CPU and CUDA; auto-detects device
- Streaming via TextIteratorStreamer
- Configurable model name for easy swap to larger variants
"""
import os
import logging
import threading
from typing import Iterator, List, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy imports — only load heavy libs when needed
_pipeline = None
_tokenizer = None
_streamer = None


MODEL_ID = os.environ.get("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Generation defaults (can be overridden per-call)
DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


def _load_model():
    """Lazy-load model. Called once on first inference."""
    global _pipeline, _tokenizer
    if _pipeline is not None:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        logger.info("Loading model: %s", MODEL_ID)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to(device)

        _pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=_tokenizer,
            device=0 if device == "cuda" else -1,
        )
        logger.info("Model loaded on %s", device)

    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}. Run: pip install transformers torch"
        ) from e


def format_messages_as_prompt(messages: List[Dict]) -> str:
    """
    Convert OpenAI-style messages to Qwen2.5 chat template format.
    Falls back to manual formatting if tokenizer.apply_chat_template unavailable.
    """
    _load_model()
    try:
        # Qwen2.5 tokenizer supports apply_chat_template
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text
    except Exception:
        # Fallback: manual formatting
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def generate(
    messages: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    """
    Run inference on a list of chat messages.
    Returns the assistant reply as a string.
    """
    _load_model()

    prompt = format_messages_as_prompt(messages)

    gen_kwargs = {
        **DEFAULT_GEN_KWARGS,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    try:
        outputs = _pipeline(
            prompt,
            return_full_text=False,
            **gen_kwargs,
        )
        response = outputs[0]["generated_text"].strip()
        # Strip any trailing special tokens
        for stop_token in ["<|im_end|>", "<|endoftext|>", "<eos>"]:
            if stop_token in response:
                response = response[: response.index(stop_token)].strip()
        return response
    except Exception as e:
        logger.error("Model inference failed: %s", e)
        raise


def generate_streaming(messages: List[Dict], max_new_tokens: int = 512) -> Iterator[str]:
    """
    Streaming generator — yields token chunks as they're produced.
    Uses TextIteratorStreamer for real-time output.
    """
    _load_model()
    try:
        import torch
        from transformers import TextIteratorStreamer
    except ImportError:
        # Fallback to non-streaming
        yield generate(messages, max_new_tokens=max_new_tokens)
        return

    prompt = format_messages_as_prompt(messages)
    inputs = _tokenizer(prompt, return_tensors="pt")

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    thread = threading.Thread(target=_pipeline.model.generate, kwargs=gen_kwargs)
    thread.start()

    buffer = ""
    for chunk in streamer:
        buffer += chunk
        # Stop at end tokens
        for stop in ["<|im_end|>", "<|endoftext|>"]:
            if stop in buffer:
                buffer = buffer[: buffer.index(stop)]
                yield buffer
                return
        yield buffer

    thread.join()


def get_model_info() -> Dict:
    return {
        "model_id": MODEL_ID,
        "type": "open-source",
        "provider": "Hugging Face / Qwen Team",
        "params": "0.5B",
        "license": "Apache 2.0",
    }
