from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    # add_messages merges incoming messages into the list instead of replacing it
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str

    # ── Routing ───────────────────────────────────────────────────────────────
    # The supervisor reads pipeline_stage to decide which agent runs next
    pipeline_stage: str  # intake | search | policy | qualification | scheduling | application | decision | complete | escalated
    previous_stage: Optional[str]  # so policy FAQ agent knows where to return

    # Tracks how many times each agent has been called — used to catch stuck loops
    visit_counts: dict

    # ── Lead Intake fields (populated by lead_intake agent) ───────────────────
    prospect_name: Optional[str]
    prospect_email: Optional[str]
    bedrooms: Optional[int]
    rent_max: Optional[float]
    move_in_date: Optional[str]
    zip_codes: Optional[list]
    amenities: Optional[list]
    has_pets: Optional[bool]
    needs_parking: Optional[bool]

    # ── Property Search fields (populated by property_search agent) ───────────
    shortlisted_properties: Optional[list]   # top-10 results with thumbnail_url
    selected_property_id: Optional[str]

    # ── Qualification fields (populated by qualification agent) ───────────────
    monthly_income: Optional[float]
    employment_status: Optional[str]
    rental_history: Optional[str]
    credit_consent: Optional[bool]
    ssn_token: Optional[str]        # opaque token — the raw SSN never enters state
    credit_score: Optional[int]
    income_verified: Optional[bool]

    # ── Scheduling fields (populated by scheduling agent) ─────────────────────
    tour_booking_id: Optional[str]
    tour_booked: Optional[bool]

    # ── Application fields (populated by application agent) ───────────────────
    application_id: Optional[str]
    application_submitted: Optional[bool]

    # ── Decision fields (populated by decision agent) ─────────────────────────
    decision: Optional[str]           # approve | conditional_approval | deny
    decision_reasoning: Optional[str]

    # ── Control flags ─────────────────────────────────────────────────────────
    # Frontend watches needs_pii_collection — when set to "ssn", it swaps the
    # chat input for a secure masked form that bypasses the LLM entirely
    needs_pii_collection: Optional[str]

    human_escalation_required: Optional[bool]
    escalation_reason: Optional[str]

    # ── Memory flags ──────────────────────────────────────────────────────────
    is_returning_prospect: Optional[bool]
    long_term_profile_loaded: Optional[bool]

    # Rolling summary of older turns (replaces full history past threshold)
    conversation_summary: Optional[str]
