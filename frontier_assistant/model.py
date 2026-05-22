"""
Frontier model inference — Claude Sonnet via Anthropic API.
Supports tool use (calculator, datetime, unit converter).
"""
import os
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("FRONTIER_MODEL_ID", "claude-sonnet-4-5")
MAX_TOOL_ROUNDS = 5  # Prevent infinite tool loops


def _get_client():
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    return anthropic.Anthropic(api_key=api_key)


def generate(
    messages: List[Dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    use_tools: bool = True,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate a response from Claude Sonnet, handling tool calls automatically.
    Returns final assistant text response.
    """
    from tools import TOOLS, execute_tool

    client = _get_client()

    # Separate system prompt from messages
    system = system_prompt
    api_messages = [m for m in messages if m["role"] != "system"]
    if system is None:
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                break

    kwargs = {
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system:
        kwargs["system"] = system
    if use_tools:
        kwargs["tools"] = TOOLS

    # Agentic loop: handle tool use
    for _round in range(MAX_TOOL_ROUNDS):
        try:
            response = client.messages.create(**kwargs)
        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            raise

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Extract text from final response
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_blocks).strip()

        elif response.stop_reason == "tool_use" and use_tools:
            # Execute tool calls and add results to messages
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.info("Tool call: %s(%s)", block.name, block.input)
                    result = execute_tool(block.name, block.input)
                    logger.info("Tool result: %s", result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Add assistant response + tool results to conversation
            kwargs["messages"] = kwargs["messages"] + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

        else:
            # Unexpected stop reason — return whatever we have
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_blocks).strip() or "(No response generated)"

    # Fallback after max rounds
    return "I ran into an issue completing that request. Please try again."


def generate_streaming(
    messages: List[Dict],
    max_tokens: int = 1024,
):
    """Streaming generator for Claude Sonnet (no tool use in streaming mode)."""
    client = _get_client()

    system = None
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            api_messages.append(m)

    kwargs = {
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system:
        kwargs["system"] = system

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def get_model_info() -> Dict:
    return {
        "model_id": MODEL_ID,
        "type": "frontier",
        "provider": "Anthropic",
        "context_window": "200K tokens",
        "tool_use": True,
        "safety": "Constitutional AI (built-in)",
    }
