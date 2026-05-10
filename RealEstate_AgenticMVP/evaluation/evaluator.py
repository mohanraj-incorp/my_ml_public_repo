"""
Evaluation runner — scores the RAG pipeline with RAGAS and agents with Vertex AI Eval.

Usage: python -m evaluation.evaluator

Results are printed to stdout and can be piped to BigQuery for trend tracking.
"""
import asyncio
import json
from pathlib import Path

from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

from rag.retriever import hybrid_retrieve
from tools.decision_tools import apply_decision_rules


# ── RAG Evaluation ────────────────────────────────────────────────────────────

async def evaluate_rag():
    """
    Runs each RAG golden case through the retriever and scores with RAGAS.
    RAGAS expects: question, answer, contexts, ground_truth.
    """
    cases = json.loads(Path("evaluation/golden_dataset/rag_cases.json").read_text())
    rows = []

    for case in cases:
        chunks = await hybrid_retrieve(
            query=case["question"],
            property_id=case.get("property_id"),
            top_k=5,
        )
        # Build a simple answer by concatenating retrieved chunks
        answer = " ".join(c["text"] for c in chunks) if chunks else "No information found."
        contexts = [c["text"] for c in chunks]

        rows.append({
            "question":     case["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": " ".join(case.get("expected_answer_contains", [])),
        })

    dataset = Dataset.from_list(rows)
    scores = evaluate(
        dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
    )
    print("\n── RAG Evaluation Results ──────────────────────────────")
    print(scores)
    return scores


# ── Agent Evaluation (Decision Agent) ────────────────────────────────────────

async def evaluate_decision_agent():
    """
    Runs decision golden cases through the rule engine and checks outcomes.
    Uses the rule engine directly — no LLM needed for outcome correctness.
    """
    cases = json.loads(Path("evaluation/golden_dataset/agent_cases.json").read_text())
    decision_cases = [c for c in cases if c["agent"] == "decision"]

    correct = 0
    for case in decision_cases:
        s = case["input_state"]
        # We need rent to compute income ratio — use $1850 as default for PROP_001
        result = apply_decision_rules(
            credit_score=s["credit_score"],
            monthly_income=s["monthly_income"],
            monthly_rent=1850,
            income_verified=s["income_verified"],
            rental_history=s["rental_history"],
            employment_status=s["employment_status"],
        )
        match = result["outcome"] == case["expected_outcome"]
        correct += int(match)
        status = "✓" if match else "✗"
        print(f"  {status} {case['case_id']}: expected={case['expected_outcome']} got={result['outcome']}")

    accuracy = correct / len(decision_cases) if decision_cases else 0
    print(f"\nDecision Agent Accuracy: {accuracy:.0%} ({correct}/{len(decision_cases)})")
    return accuracy


async def run_all():
    print("Running evaluation suite...\n")
    await evaluate_rag()
    await evaluate_decision_agent()


if __name__ == "__main__":
    asyncio.run(run_all())
