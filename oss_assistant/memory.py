"""
Shared conversation memory for both assistants.
Implements a sliding window buffer — keeps the last N turns in context.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import time


@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Sliding-window conversation buffer.
    Keeps `max_turns` most recent user+assistant pairs.
    System prompt is always preserved at position 0.
    """

    SYSTEM_PROMPT = (
        "You are a helpful, honest, and harmless personal assistant. "
        "Answer questions clearly and concisely. "
        "If you don't know something, say so rather than making things up. "
        "Refuse requests that are harmful, illegal, or unethical."
    )

    def __init__(self, max_turns: int = 10, system_prompt: Optional[str] = None):
        self.max_turns = max_turns
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self._messages: List[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        # Trim to max_turns (each turn = 1 user + 1 assistant message = 2 messages)
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def get_messages(self, include_system: bool = True) -> List[Dict]:
        """Return messages formatted for API consumption."""
        messages = []
        if include_system:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(m.to_dict() for m in self._messages)
        return messages

    def get_history_for_display(self) -> List[tuple]:
        """Return (user_msg, assistant_msg) pairs for Gradio Chatbot."""
        pairs = []
        user_msg = None
        for m in self._messages:
            if m.role == "user":
                user_msg = m.content
            elif m.role == "assistant" and user_msg is not None:
                pairs.append((user_msg, m.content))
                user_msg = None
        return pairs

    def clear(self) -> None:
        self._messages = []

    def export_json(self) -> str:
        return json.dumps([m.to_dict() for m in self._messages], indent=2)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._messages)//2}, max={self.max_turns})"
