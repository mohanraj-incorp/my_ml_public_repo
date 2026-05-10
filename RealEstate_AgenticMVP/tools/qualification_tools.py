"""
Qualification tools — credit check and income verification.
Both are mocked for the POC. In production these call real bureau APIs.

The credit check tool accepts an SSN token (never the raw SSN).
The token was issued by the PII Tokenization Service running on Cloud Run
and is resolved to the actual SSN only inside the credit bureau API call.
"""
import hashlib
import random


async def run_credit_check(ssn_token: str) -> dict:
    """
    Calls the credit bureau API with an opaque SSN token.
    Returns only the credit score and risk band — the raw SSN never surfaces here.

    POC: we deterministically derive a score from the token so results are
    consistent across calls for the same token (simulates idempotency).
    """
    # Derive a stable score from the token so repeated calls return the same result
    token_hash = int(hashlib.md5(ssn_token.encode()).hexdigest(), 16)
    credit_score = 580 + (token_hash % 250)   # score between 580 and 830

    if credit_score >= 720:
        risk_band = "low"
    elif credit_score >= 650:
        risk_band = "medium"
    else:
        risk_band = "high"

    return {
        "credit_score": credit_score,
        "risk_band": risk_band,
        "bureau": "mock_bureau",
    }


async def verify_income(monthly_income: float, employment_status: str) -> dict:
    """
    Verifies income against stated employment status.
    POC: applies simple business rules instead of calling a real verification API.
    """
    if employment_status == "unemployed":
        return {"verified": False, "reason": "No active employment"}

    if employment_status == "self_employed" and monthly_income < 3000:
        return {"verified": False, "reason": "Stated income below minimum for self-employed applicants"}

    # In production this would call The Work Number or similar service
    return {"verified": True, "reason": "Income consistent with employment status"}
