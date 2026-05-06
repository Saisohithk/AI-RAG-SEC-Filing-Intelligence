"""
src/ui/charts.py — High-fidelity interactive charts built with Plotly.

Why Plotly over st.bar_chart?
  - Full theming control (matches our dark UI palette exactly)
  - Hover tooltips with formatted values
  - Animations and layout control (margins, fonts, axis labels)
  - Donut/pie/waterfall chart types unavailable in st.bar_chart

All charts use the SEC_THEME config so colours and typography are consistent
across every visualisation in the app without per-chart style boilerplate.
"""

from __future__ import annotations

import streamlit as st

from src.core.schemas import LatencyBreakdown, Source

# ---------------------------------------------------------------------------
# Brand palette — matches .streamlit/config.toml and custom CSS
# ---------------------------------------------------------------------------

BRAND_PRIMARY = "#4f8ef7"
BRAND_BG      = "#080d14"
BRAND_SURFACE = "#0a111e"
BRAND_TEXT    = "#e8eaf0"
BRAND_MUTED   = "#6b7a99"

TICKER_COLORS: dict[str, str] = {
    "AAPL":  "#555555",
    "MSFT":  "#00a4ef",
    "GOOGL": "#4285f4",
    "AMZN":  "#ff9900",
    "NVDA":  "#76b900",
}

# Plotly layout defaults applied to every figure
_BASE_LAYOUT = dict(
    paper_bgcolor=BRAND_BG,
    plot_bgcolor=BRAND_SURFACE,
    font=dict(family="Inter, sans-serif", color=BRAND_TEXT, size=12),
    margin=dict(l=8, r=8, t=32, b=8),
    showlegend=False,
)

SEC_STAGE_COLORS = {
    "Retrieve": "#4f8ef7",
    "Rerank":   "#a78bfa",
    "Generate": "#34d399",
}


def render_latency_chart(latency: LatencyBreakdown) -> None:
    """
    Render a horizontal Plotly bar chart breaking down pipeline latency.

    Visual design:
      - Three colour-coded bars: Retrieve (blue), Rerank (purple), Generate (green)
      - Hover shows exact milliseconds
      - Total shown as a caption above the chart
      - Height fixed at 140px — compact, not attention-grabbing
    """
    import plotly.graph_objects as go

    stages = latency.stage_dict          # {"Retrieve": ms, "Rerank": ms, "Generate": ms}
    if not any(stages.values()):
        return

    labels = list(stages.keys())
    values = list(stages.values())
    colors = [SEC_STAGE_COLORS.get(lbl, BRAND_PRIMARY) for lbl in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}ms" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=BRAND_TEXT),
            hovertemplate="%{y}: <b>%{x:.1f}ms</b><extra></extra>",
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=140,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
        ),
    )

    st.caption(f"⏱ Pipeline total: **{latency.total_ms:.0f}ms**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_source_distribution(sources: list[Source]) -> None:
    """
    Render a donut chart showing how retrieved chunks are distributed by ticker.

    Only rendered when sources span more than one company — a single-ticker
    answer doesn't benefit from a distribution chart.
    """
    import plotly.graph_objects as go

    if not sources:
        return

    # Count chunks per ticker
    counts: dict[str, int] = {}
    for src in sources:
        counts[src.ticker] = counts.get(src.ticker, 0) + 1

    if len(counts) < 2:
        return  # Not interesting for a single company

    tickers = list(counts.keys())
    values  = list(counts.values())
    colors  = [TICKER_COLORS.get(t, BRAND_MUTED) for t in tickers]

    fig = go.Figure(
        go.Pie(
            labels=tickers,
            values=values,
            hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
            textfont=dict(size=11, color=BRAND_TEXT),
            hovertemplate="<b>%{label}</b>: %{value} chunk(s)<extra></extra>",
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=200,
        annotations=[
            dict(
                text=f"{len(sources)}<br><span style='font-size:10px'>chunks</span>",
                x=0.5, y=0.5,
                font=dict(size=14, color=BRAND_TEXT),
                showarrow=False,
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_relevance_scores(sources: list[Source]) -> None:
    """
    Render a horizontal bar chart of BGE reranker scores for each source chunk.

    Safe to call from inside OR outside an st.expander — no nested expander is
    created here. The chart renders inline wherever it is called.
    """
    import plotly.graph_objects as go

    if not sources:
        return

    labels = [
        f"{s.ticker} {s.filing_date} (#{i+1})"
        for i, s in enumerate(sources)
    ]
    scores = [s.rerank_score for s in sources]
    colors = [
        "#34d399" if sc >= 0.7 else "#fbbf24" if sc >= 0.4 else "#f87171"
        for sc in scores
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{sc:.2f}" for sc in scores],
            textposition="outside",
            textfont=dict(size=10, color=BRAND_TEXT),
            hovertemplate="%{y}<br>Score: <b>%{x:.3f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        height=max(120, len(sources) * 36),
        xaxis=dict(
            range=[0, 1.05],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=10),
            autorange="reversed",
        ),
    )

    st.markdown(
        '<p style="font-size:11px;font-weight:600;color:#6b7a99;'
        'letter-spacing:0.6px;text-transform:uppercase;margin:8px 0 4px">Reranker Scores</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
