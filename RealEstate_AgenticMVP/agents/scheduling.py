"""
Scheduling agent — shows available tour slots and books one.
Uses Cloud SQL for slot availability and booking (idempotent).
"""
from anthropic import Anthropic
from langchain_core.messages import AIMessage

from graph.state import AgentState
from tools.scheduling_tools import get_available_slots, book_tour
from config.settings import settings

client = Anthropic()

SYSTEM_PROMPT = """You are a tour scheduling assistant.
Available slots are listed below. Help the prospect pick one and call book_slot.
If they want to see slots on a specific date, ask them and the system will filter.
"""

BOOK_SLOT_TOOL = {
    "name": "book_slot",
    "description": "Books the selected tour slot for the prospect.",
    "input_schema": {
        "type": "object",
        "properties": {
            "slot_id":        {"type": "string"},
            "preferred_date": {"type": "string", "description": "YYYY-MM-DD, optional filter"},
        },
        "required": ["slot_id"],
    },
}


async def scheduling_node(state: AgentState) -> dict:
    visit_counts = state.get("visit_counts", {})
    visit_counts["scheduling"] = visit_counts.get("scheduling", 0) + 1

    property_id = state.get("selected_property_id")
    slots = await get_available_slots(property_id)

    if not slots:
        return {
            "visit_counts": visit_counts,
            "messages": [AIMessage(content="There are no available tour slots right now. A leasing agent will contact you to arrange a visit.")],
            "human_escalation_required": True,
            "escalation_reason": "No tour slots available",
        }

    slots_text = "\n".join(
        f"- Slot {s['slot_id']}: {s['slot_datetime']}" for s in slots
    )

    last_message = state["messages"][-1].content if state["messages"] else ""
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=512,
        system=f"{SYSTEM_PROMPT}\n\nAvailable slots:\n{slots_text}",
        messages=[{"role": "user", "content": last_message}],
        tools=[BOOK_SLOT_TOOL],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "book_slot":
            result = await book_tour(
                slot_id=block.input["slot_id"],
                prospect_name=state.get("prospect_name", ""),
                email=state.get("prospect_email", ""),
            )
            if result["status"] in ("confirmed", "already_confirmed"):
                return {
                    "tour_booking_id": result["booking_id"],
                    "tour_booked":     True,
                    "pipeline_stage":  "application",
                    "visit_counts":    visit_counts,
                    "messages": [AIMessage(content=f"Your tour is confirmed! Booking ID: {result['booking_id']}. Now let's complete your application.")],
                }
            else:
                return {
                    "visit_counts": visit_counts,
                    "messages": [AIMessage(content="That slot was just taken. Please choose another from the list.")],
                }

    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    return {"visit_counts": visit_counts, "messages": [AIMessage(content=text)]}
