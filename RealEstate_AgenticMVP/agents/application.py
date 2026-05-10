"""
Application agent — walks the prospect through submitting their rental application.
submit_application is idempotent (keyed by session_id).
"""
from anthropic import Anthropic
from langchain_core.messages import AIMessage

from graph.state import AgentState
from tools.application_tools import submit_application
from config.settings import settings

client = Anthropic()

SYSTEM_PROMPT = """You are a leasing application assistant.
Confirm the details below with the prospect, then call submit_application.
If anything is wrong, ask them to correct it before submitting.
"""

SUBMIT_TOOL = {
    "name": "submit_application",
    "description": "Submits the rental application once the prospect confirms their details.",
    "input_schema": {
        "type": "object",
        "properties": {
            "confirmed": {"type": "boolean", "description": "Prospect confirmed all details are correct"},
        },
        "required": ["confirmed"],
    },
}


async def application_node(state: AgentState) -> dict:
    visit_counts = state.get("visit_counts", {})
    visit_counts["application"] = visit_counts.get("application", 0) + 1

    # If already submitted (idempotency — LangGraph retried this node), skip
    if state.get("application_submitted"):
        return {
            "pipeline_stage": "decision",
            "visit_counts": visit_counts,
            "messages": [AIMessage(content="Your application has already been submitted. Moving on to the decision stage.")],
        }

    summary = (
        f"Here's a summary of your application:\n"
        f"- Property: {state.get('selected_property_id')}\n"
        f"- Monthly income: ${state.get('monthly_income', 'not provided')}\n"
        f"- Employment: {state.get('employment_status', 'not provided')}\n"
        f"- Rental history: {state.get('rental_history', 'not provided')}\n\n"
        f"Shall I go ahead and submit?"
    )

    last_message = state["messages"][-1].content if state["messages"] else "yes"

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=512,
        system=f"{SYSTEM_PROMPT}\n\nApplication details:\n{summary}",
        messages=[{"role": "user", "content": last_message}],
        tools=[SUBMIT_TOOL],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_application" and block.input.get("confirmed"):
            result = await submit_application(
                session_id=state["session_id"],
                property_id=state.get("selected_property_id", ""),
                prospect_name=state.get("prospect_name", ""),
                prospect_email=state.get("prospect_email", ""),
                monthly_income=state.get("monthly_income", 0),
                employment_status=state.get("employment_status", ""),
            )
            return {
                "application_id":       result["application_id"],
                "application_submitted": True,
                "pipeline_stage":        "decision",
                "visit_counts":          visit_counts,
                "messages": [AIMessage(content=f"Application submitted! Reference: {result['application_id']}. Reviewing now...")],
            }

    text = next((b.text for b in response.content if hasattr(b, "text")), summary)
    return {"visit_counts": visit_counts, "messages": [AIMessage(content=text)]}
