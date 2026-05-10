"""
Property Search agent — translates prospect preferences into a SQL query,
returns the top-10 results, and helps the prospect select one.

The search itself is a structured SQL call — no LLM needed for that part.
Claude is used only to present results conversationally and handle follow-ups
like "show me more details" or "do you have anything cheaper?".
"""
from anthropic import Anthropic
from langchain_core.messages import AIMessage

from graph.state import AgentState
from tools.property_tools import search_properties, get_unit_details
from memory.short_term import get_context_messages
from config.settings import settings

client = Anthropic()

SYSTEM_PROMPT = """You are a property search assistant.
You have already retrieved a list of matching apartments (provided below).
Present them conversationally — highlight the best matches first.
When the prospect picks a property, call the select_property tool.
If they ask about lease policies, pets, or rules — say you'll check and let the supervisor route to the policy agent.
"""

SELECT_PROPERTY_TOOL = {
    "name": "select_property",
    "description": "Called when the prospect has chosen a specific property to proceed with.",
    "input_schema": {
        "type": "object",
        "properties": {
            "property_id": {"type": "string"},
        },
        "required": ["property_id"],
    },
}


async def property_search_node(state: AgentState) -> dict:
    visit_counts = state.get("visit_counts", {})
    visit_counts["property_search"] = visit_counts.get("property_search", 0) + 1

    # Run the structured SQL search using fields collected during intake
    properties = await search_properties(
        bedrooms=state.get("bedrooms"),
        rent_max=state.get("rent_max"),
        zip_codes=state.get("zip_codes"),
        amenities=state.get("amenities"),
        move_in_date=state.get("move_in_date"),
    )

    if not properties:
        return {
            "visit_counts": visit_counts,
            "shortlisted_properties": [],
            "messages": [AIMessage(content="I couldn't find any available properties matching your criteria. Could you adjust your budget or location?")],
        }

    # Format results as text for Claude to present conversationally
    properties_text = "\n".join(
        f"- {p['property_id']}: {p['name']}, {p['city']} | {p['bedrooms']}BR | ${p['rent']}/mo | {', '.join(p.get('amenities') or [])}"
        for p in properties
    )

    context_messages = get_context_messages(state["messages"], state.get("conversation_summary"))
    anthropic_messages = [
        {"role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant",
         "content": m.content}
        for m in context_messages
        if m.__class__.__name__ in ("HumanMessage", "AIMessage")
    ]

    system = f"{SYSTEM_PROMPT}\n\nAvailable properties:\n{properties_text}"

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=system,
        messages=anthropic_messages,
        tools=[SELECT_PROPERTY_TOOL],
    )

    # Prospect selected a property
    for block in response.content:
        if block.type == "tool_use" and block.name == "select_property":
            selected_id = block.input["property_id"]
            details = await get_unit_details(selected_id)
            return {
                "selected_property_id": selected_id,
                "pipeline_stage": "qualification",
                "visit_counts": visit_counts,
                "shortlisted_properties": properties,
                "messages": [AIMessage(content=f"Great choice! Let me get some details about {details['name'] if details else selected_id} and we'll move on to the application process.")],
            }

    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    return {
        "shortlisted_properties": properties,
        "visit_counts": visit_counts,
        "messages": [AIMessage(content=text)],
    }
