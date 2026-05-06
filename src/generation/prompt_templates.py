"""
prompt_templates.py — SEC-domain-specific prompt templates for the RAG pipeline.

What this file does:
  Defines carefully engineered prompts that tell the LLM how to behave.
  Good prompts are the difference between hallucinated answers and accurate, cited ones.

Why domain-specific prompts matter:
  A generic "answer the question" prompt lets the LLM make things up.
  Our SEC-specific prompt:
  1. Restricts the LLM to ONLY use provided context
  2. Requires citations (ticker + date) for every claim
  3. Uses institutional finance language (not casual)
  4. Gracefully handles missing information instead of hallucinating

The three prompts here:
  1. SEC_RAG_PROMPT: Main Q&A prompt — the most important one
  2. QUERY_REWRITE_PROMPT: Generates 3 variations of the query for better retrieval
  3. CITATION_EXTRACTION_PROMPT: Extracts structured citations from answers
"""

from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# MAIN RAG PROMPT
# Used by: api/main.py to generate answers
# ============================================================

SEC_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a financial analyst AI specializing in SEC filings analysis.

Your job: Answer questions ONLY using the provided SEC filing excerpts below.

RULES YOU MUST FOLLOW:
1. Use ONLY information explicitly stated in the context — never add outside knowledge
2. Cite your source for every factual claim: format as [TICKER - FORM - DATE]
   Example: "Apple reported $383 billion in revenue [AAPL - 10-K - 2024]"
3. If the answer is not in the provided context, say exactly:
   "This information is not available in the provided SEC filings."
4. Use precise financial language: "$4.2 billion" not "4200000000"
5. Never speculate, estimate, or reason beyond what the context states
6. If multiple filings are provided, clearly distinguish which company you're citing

CONTEXT FROM SEC FILINGS:
{context}
""",
    ),
    (
        "human",
        "{question}",
    ),
])


# ============================================================
# QUERY REWRITING PROMPT
# Used by: api/main.py to expand queries for better retrieval
# ============================================================
# Why query rewriting?
# Users often phrase questions differently than how filings are written.
# "How much money did Apple make?" → "Apple total revenue", "AAPL net sales", etc.
# Generating multiple phrasings catches more relevant chunks.

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a financial search query optimizer.
Given a question about SEC filings, generate 3 alternative search queries
that would help retrieve relevant passages from 10-K annual reports.

Output ONLY a JSON array of 3 strings. No explanation, no other text.

Focus on:
- Using official SEC/financial terminology
- Including ticker symbols if company is mentioned
- Covering different aspects of the question

Example:
Input: "What is Apple's revenue?"
Output: ["Apple total revenue annual", "AAPL net sales 10-K", "Apple revenue recognition fiscal year"]
""",
    ),
    (
        "human",
        "Generate 3 search queries for: {question}",
    ),
])


# ============================================================
# CITATION EXTRACTION PROMPT
# Used by: api/main.py to extract structured citations from answers
# ============================================================

CITATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You extract structured citations from financial analysis answers.

Given an answer and the source chunks it was based on, output a JSON array of citations.

Each citation must be:
{{
  "ticker": "AAPL",
  "form_type": "10-K",
  "date": "2024",
  "quote": "exact short quote from the source (max 100 chars)"
}}

Output ONLY the JSON array. No other text.
""",
    ),
    (
        "human",
        """Answer: {answer}

Source chunks:
{source_chunks}

Extract citations as JSON array:""",
    ),
])


def format_context(documents: list) -> str:
    """
    Format retrieved documents into a context string for the prompt.

    Each chunk is labeled with its source so the LLM can cite it.
    The format matches what SEC_RAG_PROMPT expects.

    Args:
        documents: List of LangChain Documents from hybrid search + reranker

    Returns:
        Formatted string to insert into the {context} placeholder

    Example output:
        [Source: AAPL-10-K-2024 | Chunk 1]
        Apple's total net sales were $383.3 billion...

        [Source: MSFT-10-K-2024 | Chunk 2]
        Microsoft reported cloud revenue of $135.4 billion...
    """
    context_parts = []

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        ticker = doc.metadata.get("ticker", "?")
        rerank_score = doc.metadata.get("rerank_score", None)

        # Add score to context label for transparency
        score_str = f" | Relevance: {rerank_score:.2f}" if rerank_score else ""

        header = f"[Source: {source} | Chunk {i}{score_str}]"
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(context_parts)
