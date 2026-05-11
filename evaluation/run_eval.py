"""
Evaluation runner: RAGAS metrics + custom metrics over the golden dataset.

HOW TO RUN:
    python evaluation/run_eval.py

OUTPUT:
    - Console summary table
    - evaluation/results/run_<timestamp>.json  (per-sample details)
    - evaluation/baseline_metrics.json updated if --update-baseline flag passed

RAGAS METRICS USED:
  faithfulness       — is the answer supported by the retrieved context?
                       (catches hallucination systematically)
  answer_relevancy   — does the answer actually address the question?
  context_recall     — how much of the expected answer is covered by retrieved context?
  context_precision  — how much of the retrieved context is actually relevant?

CUSTOM METRICS (see evaluation/metrics.py):
  structured_accuracy — exact fact matching for Type A queries
  entity_coverage     — movie title coverage in semantic answers
  clarification_rate  — how often ambiguous queries trigger clarification

SCALE NOTE: In a CI/CD pipeline, run this as a pre-merge gate:
  pytest evaluation/ --tb=short
and fail builds where any metric drops below its stored baseline.
"""
import asyncio
import json
import logging
import os
import sys

# Ensure project root is on sys.path when running as `python evaluation/run_eval.py`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from datetime import datetime, timezone

# Load .env before any library (RAGAS, OpenAI) reads os.environ
from dotenv import load_dotenv
load_dotenv()

# Disable RAGAS telemetry — it tries to POST usage data to ragas.io and
# crashes with a DNS error when that host is unreachable.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_openai import ChatOpenAI as LangChainChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper

from agents.orchestrator import process_query, process_query_full, get_graph
from evaluation.metrics import EvalSample, compute_all_custom_metrics
from tools.sqlite_tools import init_sqlite_db
from rag.indexer import build_and_save_indexes
from memory.long_term import init_preferences_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"
BASELINE_PATH = "evaluation/baseline_metrics.json"
RESULTS_DIR = "evaluation/results"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_agent_on_sample(sample: dict, session_id: str) -> dict:
    """Run the full agent pipeline on one golden dataset sample."""
    query = sample["query"]
    result = await process_query_full(query, session_id)

    # Use actual retrieved chunks for RAGAS — the golden dataset's expected_context
    # contains placeholder keywords, not real document chunks, which causes
    # faithfulness and context_precision to always score 0.
    actual_contexts = result["context"]
    contexts = actual_contexts if actual_contexts else sample.get("expected_context", [])

    return {
        "query": query,
        "answer": result["answer"],
        "ground_truth": sample["expected_answer"],
        "contexts": contexts,
        "route": sample.get("expected_route", "semantic"),   # expected — drives metric cohorts
        "actual_route": result.get("route", ""),             # actual — drives clarification metrics
        "type": sample.get("type", "B"),
    }


async def collect_agent_responses(golden: list[dict]) -> list[dict]:
    """Run all golden samples through the agent sequentially."""
    results = []
    for i, sample in enumerate(golden):
        session_id = f"eval_session_{i}"
        logger.info(f"Evaluating sample {i+1}/{len(golden)}: {sample['query'][:60]}")
        try:
            result = await run_agent_on_sample(sample, session_id)
            results.append(result)
        except Exception as e:
            logger.error(f"Sample {i+1} failed: {e}")
            results.append({
                "query": sample["query"],
                "answer": f"ERROR: {e}",
                "ground_truth": sample["expected_answer"],
                "contexts": [],
                "route": "error",
                "type": sample.get("type", "?"),
            })
    return results


def run_ragas_evaluation(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation on semantic/RAG responses only.

    RAGAS metrics (faithfulness, context_precision, context_recall) are only
    meaningful when the answer was produced from retrieved document chunks.
    Analytical (SQL) answers have no retrieved context, so including them would
    give RAGAS empty context arrays and produce misleading near-zero scores.

    RAGAS 0.4+ requires explicit LLM and embeddings wrappers.
    We use the same OpenAI model as the rest of the system.
    """
    from config.settings import settings

    rag_results = [r for r in results if r.get("route") == "semantic"]
    if not rag_results:
        logger.warning("No semantic samples found — skipping RAGAS evaluation.")
        return {}

    logger.info(f"Running RAGAS on {len(rag_results)}/{len(results)} semantic samples (skipping analytical/SQL routes)")

    ragas_llm = LangchainLLMWrapper(
        LangChainChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            max_tokens=4096,  # llm_factory default (~3072) truncates long faithfulness JSON
        )
    )
    lc_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.openai_api_key
    )

    samples = [
        SingleTurnSample(
            user_input=r["query"],
            response=r["answer"],
            retrieved_contexts=r["contexts"] if r["contexts"] else [""],
            reference=r["ground_truth"],
        )
        for r in rag_results
    ]
    dataset = EvaluationDataset(samples=samples)

    # metrics=None uses RAGAS defaults: faithfulness, answer_relevancy,
    # context_precision, context_recall — all set with our llm/embeddings.
    ragas_result = evaluate(
        dataset,
        metrics=None,
        llm=ragas_llm,
        embeddings=lc_embeddings,
        raise_exceptions=False,
    )
    return ragas_result.to_pandas().mean(numeric_only=True).to_dict()


def run_custom_evaluation(results: list[dict]) -> dict:
    """Run custom metric computation on agent responses."""
    samples = [
        EvalSample(
            query=r["query"],
            expected_answer=r["ground_truth"],
            actual_answer=r["answer"],
            retrieved_context="\n".join(r["contexts"]),
            route=r["route"],
            actual_route=r.get("actual_route", r["route"]),
        )
        for r in results
    ]
    return compute_all_custom_metrics(samples)


def save_results(all_metrics: dict, per_sample: list[dict]) -> str:
    """Save evaluation results to a timestamped JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")

    output = {
        "timestamp": timestamp,
        "metrics": all_metrics,
        "per_sample_results": per_sample,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {path}")
    return path


def compare_to_baseline(metrics: dict) -> None:
    """Compare current metrics to stored baseline and print delta."""
    if not os.path.exists(BASELINE_PATH):
        print("\n[No baseline found - current run is the baseline]")
        return

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    print("\n-- Metric comparison vs baseline --")
    for key, value in metrics.items():
        if isinstance(value, float) and key in baseline:
            delta = value - baseline[key]
            symbol = "+" if delta >= 0 else "-"
            print(f"  {key:35s}: {value:.3f}  ({symbol}{abs(delta):.3f} vs baseline {baseline[key]:.3f})")


def main():
    # Ensure system is initialised before evaluation
    build_and_save_indexes()
    init_sqlite_db()
    asyncio.run(init_preferences_table())

    golden = load_golden_dataset()
    logger.info(f"Loaded {len(golden)} golden samples")

    print(f"\n{'='*60}")
    print("Running agent on all golden samples...")
    results = asyncio.run(collect_agent_responses(golden))

    semantic_count = sum(1 for r in results if r.get("route") == "semantic")
    print(f"\nRunning RAGAS evaluation on {semantic_count}/{len(results)} semantic samples...")
    ragas_metrics = run_ragas_evaluation(results)

    print("Running custom metrics...")
    custom_metrics = run_custom_evaluation(results)

    all_metrics = {**ragas_metrics, **custom_metrics}

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    for k, v in all_metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.3f}")
        else:
            print(f"  {k:35s}: {v}")

    compare_to_baseline(all_metrics)
    save_results(all_metrics, results)


if __name__ == "__main__":
    main()
