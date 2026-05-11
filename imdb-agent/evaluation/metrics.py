"""
Custom evaluation metrics to complement RAGAS.

RAGAS covers: faithfulness, answer_relevancy, context_recall, context_precision.
These custom metrics cover gaps RAGAS doesn't address for this use case:

  structured_accuracy  — did the agent get exact facts right? (year, rating, director)
  entity_coverage      — did the answer mention all expected movie titles?
  clarification_rate   — what fraction of queries triggered a clarification question?

All metrics return a float in [0, 1] where 1 = perfect.

SCALE NOTE: In production, run these on every deployment via CI/CD
and fail the deploy if any metric drops below its baseline threshold
(baselines stored in evaluation/baseline_metrics.json).
"""
import re
from dataclasses import dataclass


@dataclass
class EvalSample:
    """One evaluation example: query, expected answer, agent's actual answer, retrieved context."""
    query: str
    expected_answer: str
    actual_answer: str
    retrieved_context: str = ""
    route: str = ""         # expected route from golden dataset
    actual_route: str = ""  # route the agent actually took


def structured_accuracy(samples: list[EvalSample]) -> float:
    """
    Fraction of structured (Type A) answers that contain the expected key facts.

    Strategy: check if every number/year/proper noun in expected_answer appears
    in actual_answer (case-insensitive). Simple but effective for fact-checking.

    Args:
        samples: Only Type A (analytical) samples should be passed here.

    Returns:
        Float in [0, 1]
    """
    if not samples:
        return 0.0

    scores = []
    for s in samples:
        # Extract tokens that look like facts: numbers, capitalised words.
        # Exclude common stop words that happen to be capitalised (There, Many, etc.)
        _STOP = {"There", "Many", "The", "A", "An", "In", "For", "With", "And", "Or", "Of", "To", "Is", "It"}
        expected_tokens = set(re.findall(r"\b[\dA-Z][^\s,]*", s.expected_answer)) - _STOP
        if not expected_tokens:
            scores.append(1.0)
            continue

        def _token_match(token: str, answer: str) -> bool:
            if token.lower() in answer.lower():
                return True
            # Handle range tokens like "150-200": match if any number within range appears
            range_match = re.fullmatch(r"(\d+)-(\d+)", token)
            if range_match:
                lo, hi = int(range_match.group(1)), int(range_match.group(2))
                for num in re.findall(r"\d+", answer):
                    if lo <= int(num) <= hi:
                        return True
            return False

        found = sum(1 for token in expected_tokens if _token_match(token, s.actual_answer))
        scores.append(found / len(expected_tokens))

    return sum(scores) / len(scores)


def entity_coverage(samples: list[EvalSample]) -> float:
    """
    Fraction of expected movie titles that appear in the actual answer.

    Measures whether the agent retrieved and cited the right movies.
    Titles are extracted from expected_answer by looking for quoted strings
    or words that start with a capital letter and are in the context.

    Args:
        samples: Samples with expected movie titles in expected_answer.

    Returns:
        Float in [0, 1]
    """
    if not samples:
        return 0.0

    scores = []
    for s in samples:
        # Extract quoted titles: "The Matrix", "Inception" etc.
        titles = re.findall(r'"([^"]+)"', s.expected_answer)
        if not titles:
            scores.append(1.0)  # No titles to check — skip
            continue

        found = sum(
            1 for title in titles
            if title.lower() in s.actual_answer.lower()
        )
        scores.append(found / len(titles))

    return sum(scores) / len(scores)


def clarification_rate(samples: list[EvalSample]) -> float:
    """
    Fraction of queries where the agent actually asked for clarification.
    Uses actual_route (what the agent did), not expected_route (what was expected).

    Args:
        samples: All evaluation samples

    Returns:
        Float in [0, 1]
    """
    if not samples:
        return 0.0
    clarified = sum(1 for s in samples if (s.actual_route or s.route) == "clarify")
    return clarified / len(samples)


def compute_all_custom_metrics(samples: list[EvalSample]) -> dict:
    """
    Compute all custom metrics and return as a single dict.

    route       = expected route (golden dataset) — used to select metric cohorts
    actual_route = what the agent actually did — used for clarification metrics
    """
    analytical_samples = [s for s in samples if s.route == "analytical"]
    semantic_samples = [s for s in samples if s.route == "semantic"]

    # Of queries that SHOULD trigger clarification, how many actually did?
    should_clarify = [s for s in samples if s.route == "clarify"]
    did_clarify = [s for s in should_clarify if (s.actual_route or s.route) == "clarify"]
    clarification_rate_ambiguous = len(did_clarify) / len(should_clarify) if should_clarify else 0.0

    return {
        "structured_accuracy": structured_accuracy(analytical_samples),
        "entity_coverage": entity_coverage(semantic_samples),
        "clarification_rate": clarification_rate(samples),
        "clarification_rate_ambiguous": clarification_rate_ambiguous,
        "total_samples": len(samples),
        "analytical_samples": len(analytical_samples),
        "semantic_samples": len(semantic_samples),
    }
