"""
Guardrails / safety layer for the OSS assistant.

Two-stage pipeline:
  1. Input Filter  — fast keyword + regex pre-screen
  2. Output Filter — lightweight heuristic + optional LLM judge

Both stages are optional and configurable.
"""
import re
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocklists (input)
# ---------------------------------------------------------------------------

JAILBREAK_PATTERNS = [
    r"ignore (all |your |previous |prior )?(instructions|rules|constraints|guidelines)",
    r"you are now (DAN|jailbroken|unrestricted|evil|an? AI with no)",
    r"pretend (you are|to be) (a|an)? (evil|unrestricted|uncensored|unfiltered)",
    r"(write|generate|produce|create).{0,40}(malware|exploit|virus|ransomware|keylogger)",
    r"(how to|steps to|guide to).{0,30}(make|build|synthesize|create).{0,30}(bomb|explosive|weapon|drug|poison|meth|fentanyl)",
    r"bypass.{0,30}(safety|filter|restriction|moderation|censorship)",
    r"act as (if you have no|without any) (restriction|filter|safety|limit)",
    r"developer mode",
    r"do anything now",
    r"disable (your|all) (safety|ethical|moral) (guidelines|rules|filter)",
]

HARMFUL_TOPIC_PATTERNS = [
    r"\b(synthesize|manufacture|produce)\b.{0,40}\b(nerve agent|sarin|vx gas|ricin|anthrax)\b",
    r"\b(child|minor|underage)\b.{0,20}\b(sex|porn|nude|naked|exploit)\b",
    r"\bsuicide (method|how to|instructions|steps)\b",
    r"\bhow to (stalk|doxx|dox|hack into|break into)\b",
]

_JAILBREAK_RE = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
_HARMFUL_RE = [re.compile(p, re.IGNORECASE) for p in HARMFUL_TOPIC_PATTERNS]

# ---------------------------------------------------------------------------
# Output heuristics
# ---------------------------------------------------------------------------

HARMFUL_OUTPUT_SIGNALS = [
    r"(step [0-9]+|first|then|finally).{0,60}(bomb|explosive|detonate)",
    r"here (is|are) (the|some) (instructions|steps).{0,30}(how to|for).{0,30}(harm|kill|hurt)",
    r"(recipe|synthesis|formula).{0,30}(fentanyl|meth|heroin|ricin)",
]

_OUTPUT_RE = [re.compile(p, re.IGNORECASE) for p in HARMFUL_OUTPUT_SIGNALS]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    modified_output: Optional[str] = None


class Guardrails:
    """
    Safety layer for the OSS assistant.
    """

    REFUSAL_MESSAGE = (
        "I'm not able to help with that request. "
        "Please ask me something else — I'm happy to assist with questions, "
        "writing, analysis, coding, and many other topics."
    )

    def __init__(self, strict: bool = True):
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input(self, user_input: str) -> GuardrailResult:
        """Returns GuardrailResult(allowed=False) if input should be blocked."""
        for pattern in _JAILBREAK_RE:
            if pattern.search(user_input):
                logger.warning("Jailbreak pattern matched: %s", pattern.pattern)
                return GuardrailResult(
                    allowed=False,
                    reason=f"Jailbreak attempt detected",
                )

        for pattern in _HARMFUL_RE:
            if pattern.search(user_input):
                logger.warning("Harmful topic pattern matched: %s", pattern.pattern)
                return GuardrailResult(
                    allowed=False,
                    reason="Potentially harmful topic",
                )

        return GuardrailResult(allowed=True)

    def check_output(self, output: str) -> GuardrailResult:
        """Returns GuardrailResult with possibly modified output."""
        for pattern in _OUTPUT_RE:
            if pattern.search(output):
                logger.warning("Harmful output signal detected: %s", pattern.pattern)
                return GuardrailResult(
                    allowed=False,
                    reason="Output contained harmful content",
                    modified_output=self.REFUSAL_MESSAGE,
                )

        return GuardrailResult(allowed=True, modified_output=output)

    def get_refusal(self) -> str:
        return self.REFUSAL_MESSAGE

    # ------------------------------------------------------------------
    # Convenience: run full pipeline
    # ------------------------------------------------------------------

    def run(self, user_input: str, model_output: str) -> Tuple[bool, str]:
        """
        Returns (safe: bool, final_output: str).
        If safe=False, final_output is the refusal message.
        """
        input_check = self.check_input(user_input)
        if not input_check.allowed:
            return False, self.REFUSAL_MESSAGE

        output_check = self.check_output(model_output)
        if not output_check.allowed:
            return False, self.REFUSAL_MESSAGE

        return True, model_output
