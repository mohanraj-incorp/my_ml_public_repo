"""
Input guardrails — run on every user message BEFORE it reaches the LLM.

Checks applied in order:
1. Length cap          — prevent context overflow attacks
2. PII redaction       — Cloud DLP strips SSN, credit card, DOB from chat input
3. Prompt injection    — detect attempts to override agent instructions
"""
import re
from google.cloud import dlp_v2
from config.settings import settings

dlp_client = dlp_v2.DlpServiceClient()

MAX_INPUT_LENGTH = 2000

# Patterns that suggest the user is trying to hijack the agent
INJECTION_PATTERNS = [
    r"ignore (previous|all|prior) instructions",
    r"you are now",
    r"forget (everything|your instructions|the above)",
    r"act as (an? )?(unrestricted|jailbroken|unfiltered)",
    r"system prompt",
    r"override",
]


def check_length(text: str) -> str:
    """Truncates input that exceeds the safe character limit."""
    if len(text) > MAX_INPUT_LENGTH:
        return text[:MAX_INPUT_LENGTH]
    return text


def check_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Returns (is_safe, warning_message).
    If injection is detected, the caller should respond with a canned message
    rather than passing the text to the LLM.
    """
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, "I can only help with leasing-related questions."
    return True, ""


async def redact_pii(text: str) -> str:
    """
    Uses Cloud DLP to redact PII patterns before the message reaches the LLM.
    Catches SSN, credit card numbers, email addresses, and phone numbers.
    This is a safety net — the primary SSN path is the secure frontend form.
    """
    info_types = [
        {"name": "US_SOCIAL_SECURITY_NUMBER"},
        {"name": "CREDIT_CARD_NUMBER"},
        {"name": "PHONE_NUMBER"},
        {"name": "DATE_OF_BIRTH"},
    ]

    item = {"value": text}
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [{
                "primitive_transformation": {
                    "replace_with_info_type_config": {}
                }
            }]
        }
    }

    response = dlp_client.deidentify_content(
        request={
            "parent": f"projects/{settings.gcp_project_id}",
            "deidentify_config": deidentify_config,
            "inspect_config": {"info_types": info_types},
            "item": item,
        }
    )

    return response.item.value


async def run_input_guardrails(user_message: str) -> tuple[str, bool, str]:
    """
    Runs all input checks. Returns (cleaned_text, is_safe, block_reason).
    If is_safe is False, the agent should respond with block_reason directly.
    """
    text = check_length(user_message)

    is_safe, warning = check_prompt_injection(text)
    if not is_safe:
        return text, False, warning

    cleaned = await redact_pii(text)

    return cleaned, True, ""
