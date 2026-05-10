"""
Decision agent — synthesizes qualification data and makes the final leasing decision.

Design rules enforced here:
1. Cannot run without all required state fields (completeness gate)
2. Rule engine runs first — LLM only adds explanation, cannot override the outcome
3. Decision is written to the DB as an append-only record
4. Conditional approvals are flagged for human review
"""
import uuid
from anthropic import Anthropic
from langchain_core.messages import AIMessage
from sqlalchemy import text

from graph.state import AgentState
from tools.decision_tools import apply_decision_rules
from db.connection import AsyncSessionLocal
from config.settings import settings

client = Anthropic()

REQUIRED_FIELDS = ["credit_score", "monthly_income", "employment_status", "rental_history", "selected_property_id"]

SYSTEM_PROMPT = """You are a leasing decision assistant.
Based on the rule engine output below, write a clear and professional decision message for the prospect.
Do not change the outcome — only explain it in plain language.
For approvals: congratulate and explain next steps.
For conditional approvals: explain what additional documentation is needed.
For denials: be respectful and mention they can reapply after 6 months.
"""


async def decision_node(state: AgentState) -> dict:
    visit_counts = state.get("visit_counts", {})
    visit_counts["decision"] = visit_counts.get("decision", 0) + 1

    # ── Completeness gate — cannot decide without all required fields ─────────
    missing = [f for f in REQUIRED_FIELDS if not state.get(f)]
    if missing:
        return {
            "human_escalation_required": True,
            "escalation_reason": f"Decision agent missing required fields: {missing}",
            "pipeline_stage": "escalated",
            "visit_counts": visit_counts,
            "messages": [AIMessage(content="We need a bit more information before making a decision. A leasing agent will be in touch.")],
        }

    # ── Rule engine — deterministic, LLM cannot override ─────────────────────
    # Fetch rent for the selected property to compute income ratio
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("SELECT rent FROM properties WHERE property_id = :pid"),
            {"pid": state["selected_property_id"]},
        )
        row = result.mappings().first()
    monthly_rent = row["rent"] if row else 1500

    rules = apply_decision_rules(
        credit_score=state["credit_score"] or 0,
        monthly_income=state["monthly_income"],
        monthly_rent=monthly_rent,
        income_verified=state.get("income_verified", False),
        rental_history=state["rental_history"],
        employment_status=state["employment_status"],
    )

    outcome = rules["outcome"]

    # ── LLM writes the prospect-facing explanation ────────────────────────────
    rules_summary = (
        f"Outcome: {outcome}\n"
        f"Income ratio: {rules['income_ratio']}x\n"
        f"Issues: {rules['hard_issues']}\n"
        f"Flags: {rules['soft_flags']}"
    )

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": rules_summary}],
    )
    explanation = next((b.text for b in response.content if hasattr(b, "text")), "")

    # ── Append-only decision record ───────────────────────────────────────────
    decision_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("""
                INSERT INTO decisions
                    (decision_id, session_id, property_id, prospect_email,
                     outcome, reasoning, credit_score, income_ratio)
                VALUES
                    (:decision_id, :session_id, :property_id, :prospect_email,
                     :outcome, :reasoning, :credit_score, :income_ratio)
            """),
            {
                "decision_id":    decision_id,
                "session_id":     state["session_id"],
                "property_id":    state["selected_property_id"],
                "prospect_email": state.get("prospect_email", ""),
                "outcome":        outcome,
                "reasoning":      explanation,
                "credit_score":   state.get("credit_score"),
                "income_ratio":   rules["income_ratio"],
            },
        )
        await db.commit()

    return {
        "decision":          outcome,
        "decision_reasoning": explanation,
        "pipeline_stage":    "complete",
        "visit_counts":      visit_counts,
        # Conditional approvals go to human review before any offer is sent
        "human_escalation_required": outcome == "conditional_approval",
        "messages": [AIMessage(content=explanation)],
    }
