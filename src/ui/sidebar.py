"""
src/ui/sidebar.py — Branded sidebar with provider badges and visual controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import streamlit as st

from src.core.logging_config import handle_errors
from src.core.schemas import IngestRequest
from src.utils.api_client import APIClient

logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS: list[str] = [
    "What was Apple's total revenue in their latest 10-K?",
    "Compare Microsoft and Google's R&D spending",
    "What are Nvidia's main risk factors?",
    "How many business segments does Amazon operate?",
    "What is Apple's gross margin percentage?",
    "Summarise NVDA's data centre revenue growth",
]

# Provider metadata for display
_PROVIDERS = {
    "groq":   {"label": "Groq",   "emoji": "⚡", "desc": "500+ tok/s · Free tier",    "color": "#f97316"},
    "gemini": {"label": "Gemini", "emoji": "✨", "desc": "Best quality · Free tier",  "color": "#4285f4"},
    "ollama": {"label": "Ollama", "emoji": "🦙", "desc": "Local · No API key · Slow", "color": "#76b900"},
}


@dataclass(frozen=True)
class SidebarConfig:
    provider: str
    top_k: int
    streaming: bool


def render_sidebar(client: APIClient) -> SidebarConfig:
    with st.sidebar:
        # ── Branding header ──────────────────────────────────────────────────
        st.markdown("""
        <div style="
            padding: 20px 8px 16px;
            border-bottom: 1px solid #1a2235;
            margin-bottom: 16px;
        ">
            <div style="
                font-size: 18px;
                font-weight: 800;
                color: #e8eaf0;
                letter-spacing: -0.3px;
            ">⚙️ Controls</div>
            <div style="font-size: 11px; color: #4f6080; margin-top: 3px;">
                Configure your query pipeline
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── LLM Provider ─────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#6b7a99;'
            'letter-spacing:0.8px;text-transform:uppercase;margin:0 0 8px">LLM Provider</p>',
            unsafe_allow_html=True
        )
        provider = st.selectbox(
            "LLM Provider",
            options=list(_PROVIDERS.keys()),
            index=0,
            label_visibility="collapsed",
        )

        # Render provider info badge
        p = _PROVIDERS[provider]
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px solid #1a2235;
            border-left: 3px solid {p['color']};
            border-radius: 8px;
            padding: 8px 12px;
            margin: 6px 0 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="font-size:18px">{p['emoji']}</span>
            <div>
                <div style="font-size:13px;font-weight:600;color:#e8eaf0">{p['label']}</div>
                <div style="font-size:11px;color:#6b7a99">{p['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-K slider ─────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#6b7a99;'
            'letter-spacing:0.8px;text-transform:uppercase;margin:0 0 4px">Context Chunks</p>',
            unsafe_allow_html=True
        )
        top_k = st.slider(
            "top_k",
            min_value=1, max_value=10,
            value=st.session_state.get("top_k", 5),
            label_visibility="collapsed",
            help="Number of filing chunks sent to the LLM as context.",
        )
        st.session_state["top_k"] = top_k
        st.markdown(
            f'<p style="font-size:11px;color:#4f6080;margin:-4px 0 10px">'
            f'Top {top_k} chunks from {top_k * 4} candidates after rerank</p>',
            unsafe_allow_html=True
        )

        # ── Streaming toggle ──────────────────────────────────────────────────
        streaming = st.toggle(
            "🌊 Stream response",
            value=st.session_state.get("streaming", True),
        )
        st.session_state["streaming"] = streaming

        # ── Divider ──────────────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        # ── Sample questions ──────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#6b7a99;'
            'letter-spacing:0.8px;text-transform:uppercase;margin:0 0 8px">Try These</p>',
            unsafe_allow_html=True
        )
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"sq_{hash(q)}"):
                st.session_state["pending_question"] = q
                st.rerun()

        st.divider()

        # ── Ingest ────────────────────────────────────────────────────────────
        with st.expander("📥 Ingest New Filings"):
            with st.form("ingest_form", clear_on_submit=True):
                ticker_input = st.text_input(
                    "Tickers (comma-separated)",
                    placeholder="AAPL, MSFT, NVDA",
                )
                num_filings = st.number_input(
                    "Years of filings", min_value=1, max_value=5, value=2, step=1,
                )
                submitted = st.form_submit_button("🚀 Start Ingestion", type="primary", use_container_width=True)

            if submitted:
                _handle_ingest(client, ticker_input, int(num_filings))

        # ── Health check button ───────────────────────────────────────────────
        if st.button("🔍 Check API Health", use_container_width=True):
            _render_health(client)

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="
            padding: 16px 8px 8px;
            border-top: 1px solid #1a2235;
            margin-top: 8px;
        ">
            <div style="font-size:10px; color:#2d3a50; line-height:1.6;">
                SEC Filing Intelligence<br>
                Hybrid RAG · BGE Reranker<br>
                LangChain · Pinecone · RAGAS
            </div>
        </div>
        """, unsafe_allow_html=True)

    return SidebarConfig(provider=provider, top_k=top_k, streaming=streaming)


# ── Private helpers ──────────────────────────────────────────────────────────

def _handle_ingest(client: APIClient, ticker_input: str, num_filings: int) -> None:
    if not ticker_input.strip():
        st.warning("Enter at least one ticker symbol.")
        return
    try:
        req = IngestRequest(tickers=ticker_input.split(","), num_filings=num_filings)
    except Exception as exc:
        st.error(f"Invalid input: {exc}")
        return
    try:
        result = client.ingest(req)
        st.success(
            f"✅ Ingestion started for **{', '.join(req.tickers)}**\n\n"
            f"Job: `{result.get('job_id', 'n/a')}`"
        )
    except RuntimeError as exc:
        st.error(str(exc))


@handle_errors(fallback=None, log_level="warning")
def _render_health(client: APIClient) -> None:
    status = client.health_check()
    if status.is_healthy:
        models = ", ".join(status.models_available) or "none"
        st.success(f"✅ API healthy\n\nModels: **{models}**\nStore: **{status.vector_store}**")
    else:
        st.error("❌ API offline — start with:\n```\nuvicorn api.main:app --reload\n```")
