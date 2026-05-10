"""
Lead Intake agent — collects the prospect's apartment preferences.

Uses Claude with a tool called 'submit_intake' that Claude calls when it has
gathered all required fields. This is cleaner than parsing signal tokens from
the response text — the tool call IS the completion signal.
"""
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from graph.state import AgentState
from memory.short_term import get_context_messages
from config.settings import settings

llm = ChatAnthropic(model=settings.anthropic_model, temperature=0)

SYSTEM_PROMPT = """You are a friendly leasing assistant for RealPage properties.
Your job is to learn what the prospect is looking for in an apartment.

Collect these fields (one or two at a time — be conversational, not robotic):
- Full name and email address
- Number of bedrooms and max monthly budget
- Preferred move-in date
- Preferred zip codes or neighborhoods
- Pets (yes/no) and parking needs (yes/no)

When you have all required fields (name, email, bedrooms, rent_max, move_in_date),
call the submit_intake tool with the collected data.
If the prospect is returning and preferences are pre-filled, confirm them first.
"""

# Tool definition — Claude calls this when intake is complete
SUBMIT_INTAKE_TOOL = {
    "name": "submit_intake",
    "description": "Submit the collected prospect preferences to move to property search.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prospect_name":  {"type": "string"},
            "prospect_email": {"type": "string"},
            "bedrooms":       {"type": "integer"},
            "rent_max":       {"type": "number"},
            "move_in_date":   {"type": "string", "description": "YYYY-MM-DD format"},
            "zip_codes":      {"type": "array", "items": {"type": "string"}},
            "has_pets":       {"type": "boolean"},
            "needs_parking":  {"type": "boolean"},
        },
        "required": ["prospect_name", "prospect_email", "bedrooms", "rent_max", "move_in_date"],
    },
}


async def lead_intake_node(state: AgentState) -> dict:
    from anthropic import Anthropic
    client = Anthropic()

    context_messages = get_context_messages(
        state["messages"], state.get("conversation_summary")
    )

    # Convert LangChain messages to Anthropic API format
    anthropic_messages = [
        {"role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant",
         "content": m.content}
        for m in context_messages
        if m.__class__.__name__ in ("HumanMessage", "AIMessage")
    ]

    # Pre-fill note for returning prospects
    system = SYSTEM_PROMPT
    if state.get("is_returning_prospect"):
        system += "\nNote: This is a returning prospect — their preferences are already loaded. Confirm before proceeding."

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=system,
        messages=anthropic_messages,
        tools=[SUBMIT_INTAKE_TOOL],
    )

    # Check if Claude called the submit_intake tool (intake is complete)
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_intake":
            data = block.input
            # Update visit count for this agent
            visit_counts = state.get("visit_counts", {})
            visit_counts["lead_intake"] = visit_counts.get("lead_intake", 0) + 1

            return {
                **data,
                "pipeline_stage": "search",
                "visit_counts": visit_counts,
                "messages": [AIMessage(content="Great, I have everything I need! Let me find some properties for you.")],
            }

    # Claude is still collecting — extract text response and return
    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    visit_counts = state.get("visit_counts", {})
    visit_counts["lead_intake"] = visit_counts.get("lead_intake", 0) + 1

    return {
        "visit_counts": visit_counts,
        "messages": [AIMessage(content=text)],
    }
