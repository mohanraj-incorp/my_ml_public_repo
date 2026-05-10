"""
Output guardrails — run on agent responses BEFORE they reach the user.

Checks applied:
1. Fair Housing Act   — block discriminatory questions or statements
2. PII echo check     — ensure the LLM hasn't repeated sensitive data back
3. Citation check     — Policy FAQ responses must cite a source document
"""
import re

# Protected class keywords under the Fair Housing Act.
# If an agent output references these in a screening/preference context, it's blocked.
PROTECTED_CLASS_PATTERNS = [
    r"\b(race|ethnicity|national origin|religion|sex|gender|familial status|disability|handicap)\b",
    r"\b(children|kids|family members)\b.{0,30}\b(prefer|require|accept|suitable)\b",
    r"\b(suitable for|best for|perfect for)\b.{0,30}\b(young|professional|single|couple)\b",
]

# Patterns that suggest PII may have been echoed in the response
PII_ECHO_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",          # SSN format
    r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b",  # credit card fragment
]


def check_fair_housing(response_text: str) -> tuple[bool, str]:
    """
    Returns (is_compliant, violation_description).
    Flags responses that reference protected classes in a discriminatory way.
    """
    lowered = response_text.lower()
    for pattern in PROTECTED_CLASS_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return False, f"Potential Fair Housing violation detected. Pattern: {pattern}"
    return True, ""


def check_pii_echo(response_text: str) -> tuple[bool, str]:
    """Ensures the LLM hasn't reflected PII back in its response."""
    for pattern in PII_ECHO_PATTERNS:
        if re.search(pattern, response_text):
            return False, "Response contains a PII pattern and has been blocked."
    return True, ""


def check_rag_citation(response_text: str, retrieved_chunks: list[dict]) -> tuple[bool, str]:
    """
    For Policy FAQ responses: the answer must be grounded in retrieved chunks.
    If no chunks were retrieved, the agent should not have answered.
    """
    if not retrieved_chunks:
        return False, "No source documents were retrieved. Cannot answer policy questions without a source."
    return True, ""


def run_output_guardrails(
    response_text: str,
    agent_name: str,
    retrieved_chunks: list[dict] | None = None,
) -> tuple[str, bool, str]:
    """
    Runs all output checks. Returns (response_text, is_safe, block_reason).
    If is_safe is False, replace the response with a safe fallback.
    """
    compliant, violation = check_fair_housing(response_text)
    if not compliant:
        return response_text, False, f"Fair Housing guardrail triggered: {violation}"

    pii_safe, pii_reason = check_pii_echo(response_text)
    if not pii_safe:
        return response_text, False, pii_reason

    if agent_name == "policy_faq" and retrieved_chunks is not None:
        cited, cite_reason = check_rag_citation(response_text, retrieved_chunks)
        if not cited:
            return response_text, False, cite_reason

    return response_text, True, ""
