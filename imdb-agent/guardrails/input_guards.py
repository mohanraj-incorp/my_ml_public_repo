"""
Input guardrails: validate user messages before agent processing.

Three checks applied in sequence (cheapest first):
  1. Length check    — O(1), no LLM call
  2. Injection check — O(n) regex, no LLM call
  3. Topic relevance — LLM call, only reached if checks 1 & 2 pass

WHY THREE LAYERS: Different attack vectors need different defences.
Injection is a security concern (regex is fast and cheap).
Topic relevance prevents off-domain abuse (requires semantic understanding).

SCALE NOTE: Replace the LLM topic relevance check with a fine-tuned
DistilBERT binary classifier for <5ms inference at <1% of the LLM cost.
The InputGuardError exception interface stays identical.
"""
import re
import logging
from config.settings import settings
from config.prompts import RELEVANCE_CHECK_PROMPT

logger = logging.getLogger(__name__)

_GREETING_PATTERNS = re.compile(
    r"^\s*(hi+|hey+|hello+|howdy|greetings|good\s*(morning|afternoon|evening|day))"
    r"|(thank(s| you)+(\s+so\s+much)?!*)"
    r"|(you('re|\s+are)\s+(great|awesome|helpful|amazing))"
    r"|(bye+|goodbye|see\s+ya|take\s+care|cheers)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_GREETING_RESPONSES = {
    "hello": "Hello! What would you like to know about movies today?",
    "thanks": "You're welcome! Let me know if there's anything else you'd like to discover.",
    "bye": "Goodbye! Hope I helped you find something great to watch. Come back anytime!",
}

def _greeting_reply(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("thank", "great", "awesome", "helpful", "amazing")):
        return _GREETING_RESPONSES["thanks"]
    if any(w in t for w in ("bye", "goodbye", "see ya", "take care", "cheers")):
        return _GREETING_RESPONSES["bye"]
    return _GREETING_RESPONSES["hello"]


# Regex patterns for obvious prompt-injection attempts.
# Not exhaustive — a production system would use a dedicated injection classifier.
_INJECTION_PATTERNS = re.compile(
    r"ignore (all |previous |prior )?instructions"
    r"|forget (everything|all|prior)"
    r"|you are now"
    r"|pretend (you are|to be)"
    r"|system prompt"
    r"|<\|im_start\|>"   # GPT injection token
    r"|\[INST\]",         # Llama instruction token
    re.IGNORECASE,
)


class InputGuardError(Exception):
    """Raised when input fails a guardrail. Message is user-safe and shown directly."""
    pass


def check_length(text: str) -> None:
    if len(text) > settings.max_input_chars:
        raise InputGuardError(
            f"Your message is too long ({len(text)} chars). "
            f"Please keep questions under {settings.max_input_chars} characters."
        )


_DESTRUCTIVE_PATTERNS = re.compile(
    r"\b(delete|drop|truncate|wipe|clear|erase|remove|destroy)\b"
    r".{0,40}"
    r"\b(movie|movies|database|db|table|data|record|records|row|rows)\b"
    r"|\b(insert|add|update|modify|alter|change|edit)\b"
    r".{0,40}"
    r"\b(database|db|table|schema|record|records|row|rows)\b",
    re.IGNORECASE,
)


def check_injection(text: str) -> None:
    if _INJECTION_PATTERNS.search(text):
        raise InputGuardError(
            "I cannot perform that action. Please ask me something about movies — "
            "recommendations, ratings, directors, or plot themes."
        )


def check_destructive(text: str) -> None:
    if _DESTRUCTIVE_PATTERNS.search(text):
        raise InputGuardError(
            "That's outside what I can do. I'm a read-only movie assistant — "
            "I can search, compare, and recommend, but I cannot modify or delete any data."
        )


async def check_topic_relevance(text: str, llm) -> None:
    """
    LLM classifier: verify the message is movie-related.
    Uses a minimal YES/NO prompt to minimise token consumption.
    Only called after the two cheap checks pass.
    """
    prompt = RELEVANCE_CHECK_PROMPT.format(message=text)
    response = await llm.ainvoke(prompt)
    if response.content.strip().upper() != "YES":
        raise InputGuardError(
            "That's outside what I can help with. I'm focused on the IMDB top-1000 — "
            "try asking about movie ratings, recommendations, directors, or plots."
        )


async def run_input_guardrails(text: str, llm) -> None:
    """
    Run all input guardrails in order. Raises InputGuardError on first failure.

    Args:
        text: Raw user input string
        llm:  LLM instance for the topic-relevance check
    """
    check_length(text)
    check_injection(text)
    check_destructive(text)
    if _GREETING_PATTERNS.search(text):
        raise InputGuardError(_greeting_reply(text))
    await check_topic_relevance(text, llm)
    logger.info(f"Input guardrails passed: '{text[:50]}'")
