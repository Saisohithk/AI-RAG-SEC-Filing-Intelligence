"""
generate_testset.py — Creates synthetic question-answer pairs from SEC filings.

What this file does:
  Uses an LLM to read chunks from actual SEC filings and generate realistic questions
  about them, along with the correct answers. These QA pairs are used to evaluate
  how well our RAG system answers financial questions.

Why synthetic test sets?
  - Real user questions are expensive to collect and label
  - LLM-generated questions cover all chunks systematically
  - Ground truth answers come directly from the source — no ambiguity

Question types generated:
  1. Factual: "What was Apple's total revenue in fiscal 2024?" → specific number
  2. Comparative: "Which had higher R&D: Apple or Microsoft?" → cross-company
  3. Risk-based: "What does Apple identify as its primary cybersecurity risk?"
  4. Segment-based: "How many operating segments does Apple have?"
"""

import json
import logging
import random
from pathlib import Path

from langchain_core.documents import Document

from src.generation.llm_router import get_llm

logger = logging.getLogger(__name__)

# Prompt that asks the LLM to generate a question FROM a given text chunk
QUESTION_GENERATION_PROMPT = """You are creating evaluation questions for a financial RAG system.

Given this excerpt from an SEC 10-K filing:
---
{chunk_text}
---
Company: {ticker}

Generate ONE factual question that:
1. Can be answered ONLY from this specific excerpt
2. Has a clear, specific answer (a number, date, name, or short fact)
3. Is the type of question a financial analyst would ask

Return ONLY a JSON object with these fields:
{{
  "question": "the question",
  "ground_truth": "the exact answer from the text",
  "difficulty": "easy/medium/hard",
  "type": "factual/risk/segment/revenue"
}}

No explanation, just JSON."""

COMPARATIVE_PROMPT = """You are creating evaluation questions for a financial RAG system.

Given these two excerpts from different companies:
---
Company A ({ticker_a}): {chunk_a}
---
Company B ({ticker_b}): {chunk_b}
---

Generate ONE comparative question that requires information from BOTH companies.
Example: "Which company had higher capital expenditure: {ticker_a} or {ticker_b}?"

Return ONLY a JSON object:
{{
  "question": "the comparative question",
  "ground_truth": "the answer comparing both companies",
  "difficulty": "hard",
  "type": "comparative"
}}

No explanation, just JSON."""


def generate_synthetic_qa(
    documents: list[Document],
    num_questions: int = 200,
    llm_provider: str = "gemini",
    output_path: str = "./evals/testset.json",
) -> list[dict]:
    """
    Generate synthetic Q&A pairs from SEC filing chunks.

    Args:
        documents: Chunked SEC filing Documents (output of SECChunker)
        num_questions: Target number of questions to generate (default: 200)
        llm_provider: LLM to use for generation (default: "gemini" for quality)
        output_path: Where to save the test set JSON file

    Returns:
        List of QA dicts: [{question, ground_truth, ticker, difficulty, type}]

    Example:
        >>> qa_pairs = generate_synthetic_qa(chunks, num_questions=50, llm_provider="gemini")
        >>> print(qa_pairs[0]["question"])
        "What was Apple's total net sales in fiscal year 2024?"
    """
    llm = get_llm(llm_provider)
    qa_pairs: list[dict] = []

    # Separate documents by ticker for comparative questions
    ticker_docs: dict[str, list[Document]] = {}
    for doc in documents:
        ticker = doc.metadata.get("ticker", "UNKNOWN")
        if ticker not in ticker_docs:
            ticker_docs[ticker] = []
        ticker_docs[ticker].append(doc)

    tickers = list(ticker_docs.keys())
    logger.info(f"Generating {num_questions} questions from {len(tickers)} companies: {tickers}")

    # How many of each type to generate
    # 70% factual, 30% comparative (comparative needs 2+ tickers)
    num_comparative = min(int(num_questions * 0.3), num_questions) if len(tickers) >= 2 else 0
    num_factual = num_questions - num_comparative

    # --- Generate factual questions ---
    # Sample chunks across all tickers proportionally
    all_docs_sample = random.sample(documents, min(num_factual, len(documents)))

    for i, doc in enumerate(all_docs_sample):
        if len(qa_pairs) >= num_factual:
            break

        try:
            # Ask LLM to generate a question from this chunk
            prompt = QUESTION_GENERATION_PROMPT.format(
                chunk_text=doc.page_content[:1000],  # Limit to 1000 chars
                ticker=doc.metadata.get("ticker", "UNKNOWN"),
            )

            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse the JSON response
            qa = json.loads(response_text.strip())
            qa["ticker"] = doc.metadata.get("ticker", "UNKNOWN")
            qa["relevant_chunk_ids"] = [doc.metadata.get("chunk_id", "")]
            qa_pairs.append(qa)

            # Log progress every 10 questions
            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {len(qa_pairs)}/{num_factual} factual questions...")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping chunk {i}: invalid LLM response — {e}")
            continue
        except Exception as e:
            logger.error(f"Error generating question for chunk {i}: {e}")
            continue

    # --- Generate comparative questions ---
    if num_comparative > 0 and len(tickers) >= 2:
        logger.info(f"Generating {num_comparative} comparative questions...")

        for i in range(num_comparative):
            try:
                # Pick two different random tickers
                ticker_a, ticker_b = random.sample(tickers, 2)

                # Pick one chunk from each
                chunk_a = random.choice(ticker_docs[ticker_a])
                chunk_b = random.choice(ticker_docs[ticker_b])

                prompt = COMPARATIVE_PROMPT.format(
                    ticker_a=ticker_a,
                    chunk_a=chunk_a.page_content[:500],
                    ticker_b=ticker_b,
                    chunk_b=chunk_b.page_content[:500],
                )

                response = llm.invoke(prompt)
                response_text = response.content if hasattr(response, "content") else str(response)

                qa = json.loads(response_text.strip())
                qa["ticker"] = f"{ticker_a},{ticker_b}"
                qa["relevant_chunk_ids"] = [
                    chunk_a.metadata.get("chunk_id", ""),
                    chunk_b.metadata.get("chunk_id", ""),
                ]
                qa_pairs.append(qa)

            except Exception as e:
                logger.warning(f"Skipping comparative question {i}: {e}")
                continue

    # Save to JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(qa_pairs, indent=2))

    logger.info(f"Test set saved: {len(qa_pairs)} questions → {output_path}")
    return qa_pairs
