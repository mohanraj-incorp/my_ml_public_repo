"""
Supervisor agent — the orchestrator of the entire pipeline.

Responsibilities:
1. Run input guardrails on every user message
2. Detect stuck loops via visit_counts
3. Load long-term memory for returning prospects (first turn only)
4. Ask Claude to determine the next pipeline stage
5. Trigger summarization when conversation history grows too long
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, AIMessage

from graph.state import AgentState
from guardrails.input_guardrails import run_input_guardrails
from memory.long_term import load_prospect_profile
from memory.short_term import maybe_summarize
from config.settings import settings

llm = ChatAnthropic(model=settings.anthropic_model, temperature=0)

# Max times any single agent can be called before we escalate to a human
AGENT_MAX_VISITS = {
    "lead_intake":     6,
    "property_search": 5,
    "policy_faq":      10,
    "qualification":   6,
    "scheduling":      4,
    "application":     8,
    "decision":        2,
}

ROUTING_PROMPT = """You are the orchestrator of a leasing pipeline.
Based on the current state and conversation, decide which pipeline stage is next.

Current stage: {stage}
Prospect name: {name}
Fields collected: {fields}

Stages in order: intake → search → qualification → scheduling → application → decision
A prospect can ask a policy question (pets, lease terms etc.) at any stage — route to 'policy' and return to the previous stage after.

Reply with ONLY one word — the next stage name.
Valid values: intake, search, policy, qualification, scheduling, application, decision, complete, escalated
"""


async def supervisor_node(state: AgentState) -> dict:
    messages = state.get("messages", [])

    # ── 1. Input guardrails ──────────────────────────────────────────────────
    if messages:
        last_msg = messages[-1].content if hasattr(messages[-1], "content") else ""
        cleaned, is_safe, block_reason = await run_input_guardrails(last_msg)
        if not is_safe:
            return {"messages": [AIMessage(content=block_reason)]}

    # ── 2. Recursion / stuck-loop detection ──────────────────────────────────
    visit_counts = state.get("visit_counts", {})
    for agent, count in visit_counts.items():
        max_visits = AGENT_MAX_VISITS.get(agent, 10)
        if count > max_visits:
            return {
                "human_escalation_required": True,
                "escalation_reason": f"{agent} called {count} times — possible stuck loop",
                "pipeline_stage": "escalated",
            }

    # ── 3. Load long-term memory on first turn for returning prospects ────────
    updates = {}
    if not state.get("long_term_profile_loaded") and state.get("prospect_email"):
        profile = await load_prospect_profile(state["prospect_email"])
        if profile:
            prefs = profile.get("preferences", {})
            updates = {
                "is_returning_prospect": True,
                "long_term_profile_loaded": True,
                "bedrooms":      prefs.get("bedrooms"),
                "rent_max":      prefs.get("rent_max"),
                "zip_codes":     prefs.get("zip_codes"),
                "has_pets":      prefs.get("has_pets"),
                "needs_parking": prefs.get("needs_parking"),
            }

    # ── 4. Summarize old turns if history is getting long ────────────────────
    new_summary = await maybe_summarize(messages, state.get("conversation_summary"))
    if new_summary:
        updates["conversation_summary"] = new_summary

    # ── 5. Ask Claude which stage to go to next ──────────────────────────────
    collected_fields = [
        f for f in ["prospect_name", "bedrooms", "rent_max", "credit_score",
                    "tour_booked", "application_submitted"]
        if state.get(f) is not None
    ]

    routing_prompt = ROUTING_PROMPT.format(
        stage=state.get("pipeline_stage", "intake"),
        name=state.get("prospect_name", "unknown"),
        fields=", ".join(collected_fields) or "none yet",
    )

    response = await llm.ainvoke([SystemMessage(content=routing_prompt)] + messages[-4:])
    next_stage = response.content.strip().lower()

    # Validate — default to current stage if Claude returns something unexpected
    valid_stages = {"intake", "search", "policy", "qualification", "scheduling", "application", "decision", "complete", "escalated"}
    if next_stage not in valid_stages:
        next_stage = state.get("pipeline_stage", "intake")

    # If we're transitioning to policy mid-flow, remember where to return
    if next_stage == "policy" and state.get("pipeline_stage") != "policy":
        updates["previous_stage"] = state.get("pipeline_stage")

    return {**updates, "pipeline_stage": next_stage}
