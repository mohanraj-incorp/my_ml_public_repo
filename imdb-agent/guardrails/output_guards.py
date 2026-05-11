"""
Output guardrails: validate agent responses before returning to the user.

Checks (applied in order):
  1. PII redaction — regex, no LLM call (IMDB data shouldn't have PII,
     but this shows the pattern for domains where it matters)
  2. Length cap    — truncate verbose responses
  3. Hallucination — LLM check that cited facts appear in retrieved context
     (only for semantic/RAG answers — skipped for pure SQL queries)

DESIGN: Guards return a cleaned string, not a binary pass/fail.
Hallucination doesn't block the response — it appends a disclaimer.
In production you might retry generation or flag for human review instead.

SCALE NOTE: Replace the LLM hallucination check with a dedicated NLI
(Natural Language Inference) model for cheaper, faster checking.
RAGAS faithfulness metric (in evaluation/) is the systematic version of this.
"""
import re
import logging
from config.settings import settings
from config.prompts import HALLUCINATION_CHECK_PROMPT

logger = logging.getLogger(__name__)

_PII_RE = re.compile(
    r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"   # email
    r"|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"      # US phone
    r"|\b\d{3}-\d{2}-\d{4}\b",             # SSN-like pattern
    re.IGNORECASE,
)


def check_pii(text: str) -> str:
    cleaned = _PII_RE.sub("[REDACTED]", text)
    if cleaned != text:
        logger.warning("PII pattern detected and redacted in output")
    return cleaned


def check_output_length(text: str) -> str:
    if len(text) > settings.max_output_chars:
        logger.warning(f"Output truncated: {len(text)} → {settings.max_output_chars} chars")
        return text[: settings.max_output_chars] + "\n\n_[Response truncated for length.]_"
    return text


async def check_hallucination(answer: str, context: str, llm) -> bool:
    """
    Ask the LLM whether the answer contains claims not grounded in context.
    Returns True if potential hallucination detected.

    This is a lightweight single-call check. The systematic version is
    RAGAS faithfulness score in evaluation/run_eval.py.
    """
    if not context:
        return False  # No retrieved context to check against

    prompt = HALLUCINATION_CHECK_PROMPT.format(
        context=context[:2000], answer=answer[:1000]
    )
    response = await llm.ainvoke(prompt)
    detected = response.content.strip().upper() == "YES"
    if detected:
        logger.warning(f"Potential hallucination: {answer[:100]}")
    return detected


async def run_output_guardrails(
    answer: str,
    context: str = "",
    llm=None,
    skip_hallucination: bool = False,
) -> str:
    """
    Run all output guardrails. Returns the cleaned response string.

    Args:
        answer:             Raw agent response
        context:            Retrieved RAG passages (used for hallucination check)
        llm:                LLM instance for hallucination check
        skip_hallucination: True for analytical (SQL/pandas) answers with no RAG context
    """
    answer = check_pii(answer)
    answer = check_output_length(answer)

    if not skip_hallucination and llm and context:
        hallucinated = await check_hallucination(answer, context, llm)
        if hallucinated:
            answer += (
                "\n\n_Note: Some details in this response could not be fully "
                "verified against the dataset. Please cross-check critical facts._"
            )

    return answer
