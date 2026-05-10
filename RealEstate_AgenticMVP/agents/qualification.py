"""
Qualification agent — collects income, employment, rental history, and runs credit check.

The credit check flow is special:
- Agent sets needs_pii_collection = "ssn" in state
- Frontend sees this flag and renders a secure masked input (bypasses LLM)
- Frontend sends SSN to the PII Tokenization Service, gets back a token
- Frontend re-submits the conversation with the token in a system message
- Agent reads the token from state and calls run_credit_check(ssn_token)
"""
from anthropic import Anthropic
from langchain_core.messages import AIMessage, SystemMessage

from graph.state import AgentState
from tools.qualification_tools import run_credit_check, verify_income
from memory.short_term import get_context_messages
from config.settings import settings

client = Anthropic()

SYSTEM_PROMPT = """You are a leasing qualification assistant.
Collect the following from the prospect (one at a time, conversationally):
1. Monthly gross income
2. Employment status (full_time, part_time, self_employed, unemployed)
3. Rental history (clean, late_payments, eviction, first_time_renter)
4. Consent to run a credit check

Important rules:
- Do NOT ask about race, religion, national origin, family status, or disability.
- Once you have items 1-3 and credit consent, call the request_ssn tool.
- If consent is refused, call the skip_credit_check tool.
"""

REQUEST_SSN_TOOL = {
    "name": "request_ssn",
    "description": "Call this when income, employment, rental history are collected and the prospect has consented to a credit check. The frontend will render a secure SSN input.",
    "input_schema": {
        "type": "object",
        "properties": {
            "monthly_income":     {"type": "number"},
            "employment_status":  {"type": "string"},
            "rental_history":     {"type": "string"},
        },
        "required": ["monthly_income", "employment_status", "rental_history"],
    },
}

SKIP_CREDIT_CHECK_TOOL = {
    "name": "skip_credit_check",
    "description": "Call this if the prospect refuses the credit check. The application will proceed without a credit score.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


async def qualification_node(state: AgentState) -> dict:
    visit_counts = state.get("visit_counts", {})
    visit_counts["qualification"] = visit_counts.get("qualification", 0) + 1

    # If we already have an SSN token (frontend submitted it), run the credit check
    if state.get("ssn_token") and not state.get("credit_score"):
        credit_result = await run_credit_check(state["ssn_token"])
        income_result = await verify_income(
            state.get("monthly_income", 0),
            state.get("employment_status", ""),
        )
        return {
            "credit_score":    credit_result["credit_score"],
            "income_verified": income_result["verified"],
            "needs_pii_collection": None,   # clear the flag
            "pipeline_stage":  "scheduling",
            "visit_counts":    visit_counts,
            "messages": [AIMessage(
                content=f"Thanks! I've completed the qualification checks. "
                        f"Let's schedule a tour for the property you selected."
            )],
        }

    context_messages = get_context_messages(state["messages"], state.get("conversation_summary"))
    anthropic_messages = [
        {"role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant",
         "content": m.content}
        for m in context_messages
        if m.__class__.__name__ in ("HumanMessage", "AIMessage")
    ]

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=anthropic_messages,
        tools=[REQUEST_SSN_TOOL, SKIP_CREDIT_CHECK_TOOL],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "request_ssn":
            # Signal to frontend to show secure SSN form — LLM will not see the SSN
            return {
                **block.input,
                "credit_consent":       True,
                "needs_pii_collection": "ssn",
                "visit_counts":         visit_counts,
                "messages": [AIMessage(
                    content="To run the credit check, I'll need your Social Security Number. "
                            "Please enter it in the secure form below — it won't be stored in our chat."
                )],
            }

        if block.type == "tool_use" and block.name == "skip_credit_check":
            return {
                "credit_consent":  False,
                "pipeline_stage":  "scheduling",
                "visit_counts":    visit_counts,
                "messages": [AIMessage(
                    content="No problem. We'll proceed without a credit check. Let's schedule your tour!"
                )],
            }

    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    return {"visit_counts": visit_counts, "messages": [AIMessage(content=text)]}
