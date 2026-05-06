"""
run_ragas.py — Evaluates the RAG system using the RAGAS framework.

What this file does:
  Runs our full RAG pipeline on every question in the test set, then measures
  quality using 4 RAGAS metrics that cover different failure modes.

The 4 RAGAS Metrics:
  1. Faithfulness (>0.85 target):
     "Is the answer grounded in the retrieved context?"
     Low score = hallucination. The LLM is making things up.

  2. Answer Relevancy (>0.80 target):
     "Does the answer actually address the question asked?"
     Low score = the answer is off-topic or vague.

  3. Context Recall (>0.70 target):
     "Did we retrieve ALL the chunks needed to answer the question?"
     Low score = we're missing relevant documents (retrieval problem).

  4. Context Precision (>0.75 target):
     "Are the retrieved chunks actually relevant? (No noise?)"
     Low score = we're fetching irrelevant chunks (precision problem).

Why RAGAS matters for job interviews:
  Every senior AI engineer asks "how do you evaluate your RAG system?"
  Having RAGAS scores with baseline comparisons shows you think about
  quality systematically — not just that it "seems to work."
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

# RAGAS 0.4.x API — uses EvaluationDataset + SingleTurnSample instead of HuggingFace Dataset
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from src.core.config import settings
from src.generation.llm_router import get_llm
from src.generation.prompt_templates import SEC_RAG_PROMPT, format_context
from src.ingestion.embedder import JinaEmbedder, VectorStoreManager
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import BGEReranker

logger = logging.getLogger(__name__)

# Target scores — used in the results table to show PASS/FAIL
METRIC_TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.70,
    "context_precision": 0.75,
}


def run_evaluation(
    testset_path: str = "./evals/testset.json",
    llm_provider: str = "gemini",
    output_path: str = "./evals/results/",
    sample_size: int | None = None,
) -> dict:
    """
    Run the full RAGAS evaluation pipeline.

    For each question in the test set:
    1. Run hybrid search + reranker to retrieve relevant chunks
    2. Generate an answer using the LLM
    3. Collect: question, answer, retrieved_contexts, ground_truth
    Then evaluate ALL results at once using RAGAS.

    Args:
        testset_path: Path to JSON test set (from generate_testset.py)
        llm_provider: LLM to use for answer generation ("ollama", "groq", "gemini")
        output_path: Directory to save CSV results
        sample_size: If set, only evaluate this many questions (for quick testing)

    Returns:
        Dict mapping metric names to scores: {"faithfulness": 0.91, ...}

    Example:
        >>> scores = run_evaluation(llm_provider="gemini", sample_size=20)
        >>> print(scores["faithfulness"])
        0.91
    """
    logger.info(f"Starting RAGAS evaluation with provider: {llm_provider}")

    # Load test set
    testset = json.loads(Path(testset_path).read_text())
    if sample_size:
        testset = testset[:sample_size]
        logger.info(f"Using {sample_size} samples (subset of full test set)")

    logger.info(f"Evaluating {len(testset)} questions...")

    # Initialize the RAG pipeline components
    embedder = JinaEmbedder()
    vector_store = VectorStoreManager(use_local=settings.USE_LOCAL_CHROMA)
    searcher = HybridSearcher(vector_store, embedder)
    reranker = BGEReranker()
    llm = get_llm(llm_provider)

    # Run the pipeline on every test question
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],      # List of retrieved chunk texts
        "ground_truth": [],  # Expected answer from test set
    }

    for i, qa_pair in enumerate(testset):
        question = qa_pair["question"]
        ground_truth = qa_pair["ground_truth"]

        try:
            # Step 1: Retrieve relevant chunks
            candidates = searcher.search(question, top_k=settings.TOP_K_RETRIEVAL)
            top_docs = reranker.rerank(question, candidates, top_k=settings.TOP_K_RERANK)

            # Step 2: Generate answer
            context_text = format_context(top_docs)
            chain = SEC_RAG_PROMPT | llm
            response = chain.invoke({"context": context_text, "question": question})
            answer = response.content if hasattr(response, "content") else str(response)

            # Step 3: Collect for RAGAS
            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append([doc.page_content for doc in top_docs])
            eval_data["ground_truth"].append(ground_truth)

            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i+1}/{len(testset)} questions...")

        except Exception as e:
            logger.error(f"Error on question {i}: {e}")
            # Add placeholder to keep dataset aligned
            eval_data["question"].append(question)
            eval_data["answer"].append("Error generating answer")
            eval_data["contexts"].append([""])
            eval_data["ground_truth"].append(ground_truth)

    # Convert to RAGAS 0.4.x EvaluationDataset format
    # Each row becomes a SingleTurnSample with named fields
    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
            reference=gt,
        )
        for q, a, ctx, gt in zip(
            eval_data["question"],
            eval_data["answer"],
            eval_data["contexts"],
            eval_data["ground_truth"],
        )
    ]
    dataset = EvaluationDataset(samples=samples)

    # Run RAGAS evaluation with 0.4.x class-based metrics
    logger.info("Running RAGAS metrics (this may take a few minutes)...")
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
    )

    scores = {
        "faithfulness": float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_recall": float(results["context_recall"]),
        "context_precision": float(results["context_precision"]),
    }

    # Save results to CSV with timestamp
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"ragas_results_{llm_provider}_{timestamp}.csv"
    results.to_pandas().to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    print_results_table(scores)
    return scores


def compare_providers(testset_path: str = "./evals/testset.json") -> dict:
    """
    Run evaluation for all three LLM providers and compare results.

    This shows which provider performs best on SEC Q&A — useful for
    choosing the right LLM for production deployment.

    Args:
        testset_path: Path to test set JSON

    Returns:
        Dict: {provider: {metric: score}}
    """
    providers = ["ollama", "groq", "gemini"]
    comparison: dict[str, dict] = {}

    for provider in providers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating provider: {provider.upper()}")
        logger.info(f"{'='*50}")

        try:
            scores = run_evaluation(testset_path=testset_path, llm_provider=provider)
            comparison[provider] = scores
        except Exception as e:
            logger.error(f"Failed to evaluate {provider}: {e}")
            comparison[provider] = {"error": str(e)}

    # Save comparison results
    output_path = Path("./evals/results/provider_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2))

    logger.info(f"\nProvider comparison saved to {output_path}")
    return comparison


def print_results_table(results: dict) -> None:
    """
    Print a formatted ASCII table showing RAGAS metric scores vs targets.

    Args:
        results: Dict of {metric_name: score}

    Output example:
        +-----------------------+--------+--------+--------+
        | Metric                | Score  | Target | Status |
        +-----------------------+--------+--------+--------+
        | Faithfulness          | 0.912  | 0.850  |  PASS  |
        | Answer Relevancy      | 0.834  | 0.800  |  PASS  |
        | Context Recall        | 0.756  | 0.700  |  PASS  |
        | Context Precision     | 0.781  | 0.750  |  PASS  |
        +-----------------------+--------+--------+--------+
    """
    print("\n" + "=" * 56)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 56)
    print(f"  {'Metric':<22} {'Score':>7}  {'Target':>7}  {'Status':>6}")
    print("-" * 56)

    for metric_key, target in METRIC_TARGETS.items():
        score = results.get(metric_key, 0.0)
        status = "PASS" if score >= target else "FAIL"
        status_str = f"✓ {status}" if status == "PASS" else f"✗ {status}"

        # Format metric name nicely
        metric_name = metric_key.replace("_", " ").title()
        print(f"  {metric_name:<22} {score:>7.3f}  {target:>7.3f}  {status_str:>8}")

    print("=" * 56 + "\n")
