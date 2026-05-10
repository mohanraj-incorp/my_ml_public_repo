"""
Long-term memory — stores and retrieves prospect profiles in Firestore.

When a prospect returns for a new session, we load their stored preferences
so the agent doesn't ask them for information we already have.

Memory decay: preferences older than 90 days are flagged as potentially stale.
The agent asks the prospect to confirm rather than silently assuming.
"""
from datetime import datetime, timedelta
from google.cloud import firestore
from config.settings import settings

db = firestore.AsyncClient()
STALE_DAYS = 90


async def load_prospect_profile(email: str) -> dict | None:
    """
    Looks up a returning prospect by email.
    Returns their stored preferences, or None if they're a new prospect.
    """
    doc_ref = db.collection(settings.firestore_collection_prospects).document(email)
    doc = await doc_ref.get()

    if not doc.exists:
        return None

    profile = doc.to_dict()

    # Flag stale preferences so the agent confirms them rather than assuming
    updated_at = profile.get("preference_updated_at")
    if updated_at:
        age = datetime.utcnow() - datetime.fromisoformat(updated_at)
        profile["preferences_stale"] = age > timedelta(days=STALE_DAYS)
    else:
        profile["preferences_stale"] = True

    return profile


async def save_prospect_profile(email: str, profile_data: dict) -> None:
    """
    Saves or updates a prospect's profile after a session ends.
    Stores preferences, properties viewed, and application status.
    """
    doc_ref = db.collection(settings.firestore_collection_prospects).document(email)

    await doc_ref.set(
        {
            **profile_data,
            "preference_updated_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
        },
        merge=True,  # merge=True updates fields without overwriting the whole document
    )


def build_profile_from_state(state: dict) -> dict:
    """
    Extracts the fields worth persisting from the current session state.
    We only save things a returning prospect would want pre-filled.
    """
    return {
        "prospect_name": state.get("prospect_name"),
        "preferences": {
            "bedrooms": state.get("bedrooms"),
            "rent_max": state.get("rent_max"),
            "zip_codes": state.get("zip_codes"),
            "amenities": state.get("amenities"),
            "has_pets": state.get("has_pets"),
            "needs_parking": state.get("needs_parking"),
        },
        "properties_viewed": [
            p["property_id"] for p in (state.get("shortlisted_properties") or [])
        ],
        "application_status": "submitted" if state.get("application_submitted") else None,
        "last_decision": state.get("decision"),
    }
