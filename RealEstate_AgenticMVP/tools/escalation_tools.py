"""
Human escalation tool — notifies the leasing agent that a session needs attention.
In production this would trigger a Slack message, email, or CRM task.
For the POC it logs the escalation and writes it to Cloud Logging.
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


async def notify_human_agent(session_id: str, reason: str, prospect_name: str | None = None) -> dict:
    """
    Escalates the current session to a human leasing agent.
    After this call the graph routes to END — the human takes over.
    """
    escalation_record = {
        "session_id": session_id,
        "reason": reason,
        "prospect_name": prospect_name,
        "escalated_at": datetime.utcnow().isoformat(),
        "status": "pending_human_review",
    }

    # Structured log — Cloud Logging picks this up and can alert on it
    logger.warning("HUMAN_ESCALATION_REQUIRED", extra={"json_fields": escalation_record})

    # Production: POST to internal webhook / Slack / CRM API here
    return escalation_record
