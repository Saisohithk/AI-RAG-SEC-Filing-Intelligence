"""
src/ui/chat.py — Chat rendering with rich source cards and streaming support.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.core.schemas import ChatMessage, LatencyBreakdown, QueryRequest, Source
from src.ui.charts import (
    render_latency_chart,
    render_relevance_scores,
    render_source_distribution,
)
from src.utils.api_client import APIClient

logger = logging.getLogger(__name__)

TICKER_COLORS: dict[str, str] = {
    "AAPL":  "#6e6e6e",
    "MSFT":  "#00a4ef",
    "GOOGL": "#4285f4",
    "AMZN":  "#ff9900",
    "NVDA":  "#76b900",
}

TICKER_BG: dict[str, str] = {
    "AAPL":  "rgba(110,110,110,0.15)",
    "MSFT":  "rgba(0,164,239,0.12)",
    "GOOGL": "rgba(66,133,244,0.12)",
    "AMZN":  "rgba(255,153,0,0.12)",
    "NVDA":  "rgba(118,185,0,0.12)",
}


def render_message_history(messages: list[ChatMessage]) -> None:
    for msg in messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.role == "assistant":
                if msg.has_sources:
                    render_sources(msg.sources)
                if msg.has_latency:
                    render_latency_chart(msg.latency)  # type: ignore[arg-type]


def render_sources(sources: list[Source]) -> None:
    if not sources:
        return

    with st.expander(f"📎  **{len(sources)} sources** — click to inspect citations"):
        # Summary row
        tickers = sorted({s.ticker for s in sources})
        ticker_pills = " ".join(
            f'<span style="background:{TICKER_BG.get(t, "rgba(79,142,247,0.12)")}; '
            f'border:1px solid {TICKER_COLORS.get(t, "#4f8ef7")}40; '
            f'color:{TICKER_COLORS.get(t, "#4f8ef7")}; '
            f'padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600">{t}</span>'
            for t in tickers
        )
        st.markdown(
            f'<div style="margin-bottom:12px; display:flex; gap:6px; align-items:center; flex-wrap:wrap;">'
            f'<span style="font-size:12px; color:#6b7a99;">Companies cited:</span> {ticker_pills}</div>',
            unsafe_allow_html=True,
        )

        # Charts
        col_donut, col_scores = st.columns([1, 2])
        with col_donut:
            render_source_distribution(sources)
        with col_scores:
            render_relevance_scores(sources)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Source cards
        for i, src in enumerate(sources):
            color  = TICKER_COLORS.get(src.ticker, "#4f8ef7")
            bg     = TICKER_BG.get(src.ticker, "rgba(79,142,247,0.08)")
            score  = src.rerank_score
            bar_w  = int(score * 100)
            score_color = "#34d399" if score >= 0.7 else "#fbbf24" if score >= 0.4 else "#f87171"
            preview = src.chunk_text[:260].replace("<", "&lt;").replace(">", "&gt;")
            if len(src.chunk_text) > 260:
                preview += "…"

            st.markdown(f"""
<div style="
    background: #0a111e;
    border: 1px solid #1a2235;
    border-left: 3px solid {color};
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
">
    <!-- Header row -->
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="
                background:{bg};
                border:1px solid {color}50;
                color:{color};
                padding:3px 9px;
                border-radius:6px;
                font-size:12px;
                font-weight:700;
                letter-spacing:0.5px;
            ">{src.ticker}</span>
            <span style="color:#6b7a99; font-size:12px; font-weight:500;">10-K · {src.filing_date}</span>
            <span style="color:#3d4f6b; font-size:11px;">Chunk #{i+1}</span>
        </div>
        <div style="text-align:right;">
            <span style="font-size:11px; color:{score_color}; font-weight:600;">
                {score:.0%} relevance
            </span>
        </div>
    </div>
    <!-- Relevance bar -->
    <div style="
        background:#1a2235;
        border-radius:4px;
        height:3px;
        margin-bottom:10px;
    ">
        <div style="
            background: linear-gradient(90deg, {color}, {score_color});
            width:{bar_w}%;
            height:3px;
            border-radius:4px;
        "></div>
    </div>
    <!-- Text preview -->
    <div style="color:#8b9ab5; font-size:12px; line-height:1.6; font-style:italic;">
        "{preview}"
    </div>
</div>
""", unsafe_allow_html=True)


def process_question(
    question: str,
    config: "SidebarConfig",  # noqa: F821
    client: APIClient,
) -> ChatMessage:
    request = QueryRequest(
        question=question,
        provider=config.provider,
        top_k=config.top_k,
    )

    full_text = ""
    sources: list[Source] = []
    latency = LatencyBreakdown()

    with st.chat_message("assistant"):
        if config.streaming:
            full_text, sources, latency = _render_streaming(request, client)
        else:
            full_text, sources, latency = _render_blocking(request, client)

        if sources:
            render_sources(sources)
        if latency.total_ms > 0:
            render_latency_chart(latency)

    return ChatMessage(
        role="assistant",
        content=full_text,
        sources=sources,
        latency=latency,
    )


def _render_streaming(
    request: QueryRequest,
    client: APIClient,
) -> tuple[str, list[Source], LatencyBreakdown]:
    status_el = st.empty()
    text_el   = st.empty()

    status_el.markdown("""
    <div style="display:flex;align-items:center;gap:10px;color:#6b7a99;font-size:13px;">
        <span style="
            display:inline-block;width:8px;height:8px;
            background:#4f8ef7;border-radius:50%;
            animation:pulse 1.2s ease-in-out infinite;
        "></span>
        Searching SEC filings…
    </div>
    <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}</style>
    """, unsafe_allow_html=True)

    full_text = ""
    sources: list[Source] = []
    latency = LatencyBreakdown()
    first_token = True
    error_occurred = False

    def on_token(token: str) -> None:
        nonlocal full_text, first_token
        if first_token:
            status_el.empty()
            first_token = False
        full_text += token
        text_el.markdown(full_text + "▌")

    def on_done(s: list[Source], l: LatencyBreakdown) -> None:
        nonlocal sources, latency
        sources, latency = s, l

    def on_error(msg: str) -> None:
        nonlocal error_occurred
        error_occurred = True
        status_el.empty()
        text_el.error(f"⚠️ {msg}")

    client.parse_stream_response(request, on_token, on_done, on_error)

    if not error_occurred:
        text_el.markdown(full_text or "_No response received._")
    status_el.empty()

    return full_text, sources, latency


def _render_blocking(
    request: QueryRequest,
    client: APIClient,
) -> tuple[str, list[Source], LatencyBreakdown]:
    with st.spinner("Searching SEC filings…"):
        try:
            response = client.query(request)
        except RuntimeError as exc:
            st.error(f"⚠️ {exc}")
            return str(exc), [], LatencyBreakdown()
    st.markdown(response.answer)
    return response.answer, response.sources, response.latency_ms
