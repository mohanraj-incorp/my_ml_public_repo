"""
Decision rule engine — applies consistent business rules to every applicant.
Rules are deterministic and cannot be overridden by the LLM.
The Decision agent reads these results and writes the final outcome.
"""


def apply_decision_rules(
    credit_score: int,
    monthly_income: float,
    monthly_rent: float,
    income_verified: bool,
    rental_history: str,
    employment_status: str,
) -> dict:
    """
    Returns a recommended outcome and the reasons behind it.
    The LLM uses this as the basis for its final decision — it cannot
    override the rule engine, only add explanatory language.
    """
    issues = []
    flags = []

    # Rule 1: Income must be at least 3x monthly rent (industry standard)
    income_ratio = monthly_income / monthly_rent if monthly_rent else 0
    if income_ratio < 2.5:
        issues.append(f"Income ratio {income_ratio:.1f}x is below the 2.5x minimum")
    elif income_ratio < 3.0:
        flags.append(f"Income ratio {income_ratio:.1f}x is below preferred 3x — borderline")

    # Rule 2: Credit score thresholds
    if credit_score < 600:
        issues.append(f"Credit score {credit_score} is below minimum of 600")
    elif credit_score < 650:
        flags.append(f"Credit score {credit_score} is below preferred 650")

    # Rule 3: Income must be verified
    if not income_verified:
        issues.append("Income could not be verified")

    # Rule 4: Rental history
    if rental_history == "eviction":
        issues.append("Prior eviction on record")
    elif rental_history == "late_payments":
        flags.append("History of late payments noted")

    # Rule 5: Employment
    if employment_status == "unemployed":
        issues.append("No active employment")

    # Determine outcome
    if issues:
        outcome = "deny"
    elif flags:
        outcome = "conditional_approval"
    else:
        outcome = "approve"

    return {
        "outcome": outcome,
        "income_ratio": round(income_ratio, 2),
        "hard_issues": issues,    # these cause denial
        "soft_flags": flags,      # these cause conditional approval
    }
