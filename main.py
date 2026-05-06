"""
main.py — Streamlit entry point for the SEC Filing Intelligence system.

Run:
    streamlit run main.py
"""

import os

import streamlit as st

st.set_page_config(
    page_title="SEC Filing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.core.logging_config import setup_logging
from src.core.schemas import ChatMessage
from src.ui.chat import process_question, render_message_history
from src.ui.sidebar import render_sidebar
from src.utils.api_client import APIClient

setup_logging("INFO")

# ── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #080d14; }
.main .block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 1100px;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%) !important;
    border-right: 1px solid #1a2235 !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* Sidebar sample question buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid #1a2235 !important;
    color: #8b9ab5 !important;
    font-size: 12px !important;
    text-align: left !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    line-height: 1.4 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #4f8ef7 !important;
    color: #e8eaf0 !important;
    background: rgba(79,142,247,0.08) !important;
    transform: translateX(3px) !important;
}

/* Primary button (ingest submit) */
button[kind="primaryFormSubmit"],
button[kind="primary"] {
    background: linear-gradient(135deg, #4f8ef7 0%, #7c5cf7 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(79,142,247,0.3) !important;
}
button[kind="primaryFormSubmit"]:hover,
button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,142,247,0.45) !important;
}

/* Health check button */
section[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: rgba(79,142,247,0.1) !important;
    border-color: rgba(79,142,247,0.3) !important;
    color: #4f8ef7 !important;
}

/* ── Selectbox & slider ── */
div[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background: #4f8ef7 !important;
    border: 2px solid #4f8ef7 !important;
}

/* ── Toggle ── */
div[data-testid="stToggle"] label { color: #8b9ab5 !important; }

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    background: #0f1623 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}
/* User message accent */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid #4f8ef7 !important;
    background: #0c1420 !important;
}
/* Assistant message accent */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #7c5cf7 !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] {
    background: #0d1117 !important;
    border-top: 1px solid #1a2235 !important;
    padding: 12px 0 !important;
}
div[data-testid="stChatInput"] textarea {
    background: #111827 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #4f8ef7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
    outline: none !important;
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
    background: #0f1623 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    color: #8b9ab5 !important;
    font-size: 13px !important;
}
div[data-testid="stExpander"] summary:hover { color: #e8eaf0 !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: #0f1623 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
div[data-testid="stMetric"] label { color: #6b7a99 !important; font-size: 12px !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #e8eaf0 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* ── Success / Error / Info boxes ── */
div[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: #1a2235 !important; margin: 0.75rem 0 !important; }

/* ── Plotly transparent bg ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── API CLIENT ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_api_client() -> APIClient:
    # API_URL is injected by Docker Compose; falls back to localhost for local dev
    url = os.getenv("API_URL", "http://localhost:8000")
    return APIClient(base_url=url)


# ── SESSION STATE ────────────────────────────────────────────────────────────
def _init_session_state() -> None:
    for key, val in {
        "messages": [],
        "pending_question": None,
        "top_k": 5,
        "streaming": True,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()
client = get_api_client()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
config = render_sidebar(client)

# ── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d1421 0%, #111827 50%, #0d1421 100%);
    border: 1px solid #1a2235;
    border-radius: 16px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
">
  <!-- Decorative glow -->
  <div style="
    position:absolute; top:-60px; right:-60px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(79,142,247,0.12) 0%, transparent 70%);
    border-radius:50%;
  "></div>
  <div style="
    position:absolute; bottom:-40px; left:30%;
    width:150px; height:150px;
    background: radial-gradient(circle, rgba(124,92,247,0.08) 0%, transparent 70%);
    border-radius:50%;
  "></div>

  <!-- Badge -->
  <div style="margin-bottom:12px;">
    <span style="
      background: rgba(79,142,247,0.15);
      border: 1px solid rgba(79,142,247,0.3);
      color: #4f8ef7;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.8px;
      text-transform: uppercase;
    ">Production RAG · v1.0</span>
  </div>

  <!-- Title -->
  <h1 style="
    margin: 0 0 8px 0;
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #e8eaf0 0%, #4f8ef7 60%, #7c5cf7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
  ">📊 SEC Filing Intelligence</h1>

  <!-- Subtitle -->
  <p style="
    margin: 0 0 20px 0;
    color: #6b7a99;
    font-size: 14px;
    line-height: 1.6;
  ">
    Ask plain-English questions about SEC 10-K filings from Apple, Microsoft, Google, Amazon &amp; Nvidia.<br>
    <span style="color:#4f8ef7">Hybrid Search</span> ·
    <span style="color:#7c5cf7">BGE Reranker</span> ·
    <span style="color:#34d399">Cited Answers</span> ·
    <span style="color:#fbbf24">RAGAS Evaluated</span>
  </p>

  <!-- Pipeline badges -->
  <div style="display:flex; gap:8px; flex-wrap:wrap;">
    <span style="background:#0a1628; border:1px solid #1a2235; color:#8b9ab5; padding:4px 10px; border-radius:6px; font-size:11px;">
      🔍 BM25 + Vector → RRF
    </span>
    <span style="background:#0a1628; border:1px solid #1a2235; color:#8b9ab5; padding:4px 10px; border-radius:6px; font-size:11px;">
      🧠 BAAI/bge-reranker-base
    </span>
    <span style="background:#0a1628; border:1px solid #1a2235; color:#8b9ab5; padding:4px 10px; border-radius:6px; font-size:11px;">
      ⚡ Groq 500+ tok/s
    </span>
    <span style="background:#0a1628; border:1px solid #1a2235; color:#8b9ab5; padding:4px 10px; border-radius:6px; font-size:11px;">
      📐 RAGAS Evaluated
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LIVE STAT CARDS ──────────────────────────────────────────────────────────
health = client.health_check()
stats  = client.get_stats()

c1, c2, c3, c4 = st.columns(4)
with c1:
    status_icon  = "🟢" if health.is_healthy else "🔴"
    status_label = "Online" if health.is_healthy else "Offline"
    st.metric(label="API Status", value=f"{status_icon} {status_label}")
with c2:
    models = len(health.models_available)
    st.metric(label="LLM Providers", value=f"{models} / 3",
              delta="All active" if models == 3 else f"{models} active")
with c3:
    vec_count = stats.get("vector_count", 0) if isinstance(stats, dict) else 0
    st.metric(label="Indexed Vectors", value=f"{vec_count:,}")
with c4:
    store = (health.vector_store or "—").title()
    st.metric(label="Vector Store", value=store)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)


# ── DATASET INFO ──────────────────────────────────────────────────────────────
def _render_dataset_info() -> None:
    """Rich, beginner-friendly breakdown of the indexed dataset and pipeline."""

    # ── Section: What is a 10-K? ────────────────────────────────────────────
    st.markdown("""
<div style="background:#0a111e;border:1px solid #1a2235;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
  <h4 style="color:#4f8ef7;margin:0 0 10px;font-size:15px;">📄 What is a 10-K Filing?</h4>
  <p style="color:#8b9ab5;font-size:13px;line-height:1.7;margin:0;">
    A <strong style="color:#e8eaf0">10-K</strong> is the <em>annual report</em> every US public company must file
    with the SEC (Securities & Exchange Commission). It is the most comprehensive document a company produces —
    typically <strong style="color:#e8eaf0">100–300 pages</strong> of audited financials, risk disclosures,
    business strategy, and management discussion.<br><br>
    Think of it as the official "source of truth" for a company's financial health — far more detailed than a press
    release or earnings call. Analysts, investors, and regulators use 10-Ks to make billion-dollar decisions.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Section: Indexed companies ──────────────────────────────────────────
    st.markdown("""
<div style="background:#0a111e;border:1px solid #1a2235;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
  <h4 style="color:#4f8ef7;margin:0 0 14px;font-size:15px;">🏢 Indexed Companies & Filings</h4>
  <p style="color:#6b7a99;font-size:12px;margin:0 0 14px;">
    The system has indexed <strong style="color:#e8eaf0">15 annual 10-K filings</strong> across
    <strong style="color:#e8eaf0">5 companies</strong>, split into
    <strong style="color:#e8eaf0">4,779 searchable chunks</strong> (~319 chunks / filing).
  </p>
""", unsafe_allow_html=True)

    # Company table
    companies = [
        ("AAPL",  "🍎", "Apple Inc.",            "#6e6e6e", "Consumer Electronics · Software · Services",       "2023 · 2024 · 2025"),
        ("MSFT",  "🪟", "Microsoft Corporation", "#00a4ef", "Cloud (Azure) · Office · LinkedIn · Gaming",        "2023 · 2024 · 2025"),
        ("GOOGL", "🔍", "Alphabet Inc.",          "#4285f4", "Search · YouTube · Google Cloud · Waymo",          "2023 · 2024 · 2025"),
        ("AMZN",  "📦", "Amazon.com Inc.",        "#ff9900", "E-Commerce · AWS · Advertising · Prime",           "2023 · 2024 · 2025"),
        ("NVDA",  "🎮", "NVIDIA Corporation",     "#76b900", "AI/GPU · Data Centre · Gaming · Automotive",       "2024 · 2025 · 2026"),
    ]
    for ticker, emoji, name, color, segments, years in companies:
        st.markdown(f"""
<div style="
  display:flex; align-items:flex-start; gap:14px;
  background:#060d18; border:1px solid #1a2235;
  border-left:3px solid {color};
  border-radius:10px; padding:12px 16px; margin-bottom:8px;
">
  <div style="font-size:24px;line-height:1">{emoji}</div>
  <div style="flex:1;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
      <span style="background:rgba(255,255,255,0.05);border:1px solid {color}40;color:{color};
        padding:2px 8px;border-radius:5px;font-size:12px;font-weight:700">{ticker}</span>
      <span style="color:#e8eaf0;font-size:13px;font-weight:600">{name}</span>
    </div>
    <div style="color:#6b7a99;font-size:11px;margin-bottom:4px;">📊 Segments: {segments}</div>
    <div style="color:#4f6080;font-size:11px;">📅 Fiscal years indexed: <strong style="color:#8b9ab5">{years}</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: What's inside a 10-K (data types) ──────────────────────────
    st.markdown("""
<div style="background:#0a111e;border:1px solid #1a2235;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
  <h4 style="color:#4f8ef7;margin:0 0 12px;font-size:15px;">📊 What Information Is Inside Each Filing?</h4>
  <p style="color:#6b7a99;font-size:12px;margin:0 0 12px;">
    Each 10-K is divided into standardised sections. The RAG system can answer questions about all of them:
  </p>
""", unsafe_allow_html=True)

    sections = [
        ("💰", "Financial Statements",    "#34d399", "Income statement (revenue, profit), balance sheet (assets, debt), cash flow — the numbers everyone cares about"),
        ("⚠️",  "Risk Factors",           "#f87171", "Everything that could go wrong — competition, regulation, macroeconomics, supply chain, cybersecurity"),
        ("🏗️",  "Business Description",   "#4f8ef7", "What the company actually does, how it makes money, its products, customers, and market position"),
        ("🗣️",  "MD&A",                   "#a78bfa", "Management's Discussion & Analysis — executives explain results, compare year-over-year, discuss outlook"),
        ("⚖️",  "Legal Proceedings",      "#fbbf24", "Ongoing lawsuits, regulatory investigations, and legal risks"),
        ("👥",  "Executive Compensation", "#fb923c", "CEO/CFO salary, bonuses, stock awards — useful for governance analysis"),
        ("📐",  "Segment Reporting",      "#60a5fa", "Revenue/profit broken down by business unit (e.g. AWS vs. Amazon Retail vs. Advertising)"),
    ]
    for emoji, name, color, desc in sections:
        st.markdown(f"""
<div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #0e1828;">
  <span style="font-size:18px;width:24px;text-align:center">{emoji}</span>
  <div>
    <span style="color:{color};font-size:12px;font-weight:600">{name}</span>
    <span style="color:#4f6080;font-size:12px;"> — {desc}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: Example questions ───────────────────────────────────────────
    st.markdown("""
<div style="background:#0a111e;border:1px solid #1a2235;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
  <h4 style="color:#4f8ef7;margin:0 0 12px;font-size:15px;">💡 Example Questions You Can Ask</h4>
""", unsafe_allow_html=True)

    example_qs = {
        "Financial": [
            "What was Apple's total revenue in 2024?",
            "Compare Microsoft and Amazon's operating income",
            "What is Nvidia's gross margin percentage?",
            "How much cash does Google have on its balance sheet?",
        ],
        "Risk & Strategy": [
            "What are Nvidia's top 3 risk factors?",
            "How does Apple describe competition in the iPhone market?",
            "What regulatory risks does Google mention?",
            "What does Amazon say about AWS's competitive position?",
        ],
        "Segments": [
            "How many business segments does Amazon operate?",
            "What percentage of Microsoft's revenue comes from Azure?",
            "Break down Alphabet's revenue by segment",
            "What is Nvidia's data centre revenue vs. gaming revenue?",
        ],
        "Cross-company": [
            "Compare R&D spending across all 5 companies",
            "Which company has the highest profit margin?",
            "How do Apple and Microsoft describe their AI strategy?",
        ],
    }
    cols = st.columns(2)
    for i, (category, questions) in enumerate(example_qs.items()):
        with cols[i % 2]:
            st.markdown(f'<p style="font-size:11px;font-weight:700;color:#4f8ef7;text-transform:uppercase;letter-spacing:0.6px;margin:8px 0 6px">{category}</p>', unsafe_allow_html=True)
            for q in questions:
                st.markdown(f'<p style="color:#6b7a99;font-size:12px;margin:0 0 4px;padding-left:8px;border-left:2px solid #1a2235">→ {q}</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: How the pipeline works ─────────────────────────────────────
    st.markdown("""
<div style="background:#0a111e;border:1px solid #1a2235;border-radius:12px;padding:20px 24px;">
  <h4 style="color:#4f8ef7;margin:0 0 14px;font-size:15px;">⚙️ How the Pipeline Answers Your Question</h4>
""", unsafe_allow_html=True)

    steps = [
        ("1", "#4f8ef7", "You type a question",
         "e.g. 'What was Apple's revenue in 2024?'"),
        ("2", "#60a5fa", "Hybrid Search runs in parallel",
         "BM25 finds chunks with exact keywords · Vector search finds semantically similar chunks · "
         "RRF fusion merges both lists → Top 20 candidates"),
        ("3", "#a78bfa", "BGE Reranker scores all 20 candidates",
         "Cross-encoder reads each (question, chunk) pair together — much more accurate than vector similarity alone. "
         "Returns Top 5 most relevant chunks."),
        ("4", "#34d399", "LLM generates a cited answer",
         "The 5 chunks are formatted as context and sent to your chosen LLM (Groq / Gemini / Ollama). "
         "The prompt enforces: cite every claim with [TICKER - 10-K - YEAR], never hallucinate."),
        ("5", "#fbbf24", "Sources + latency displayed",
         "You see the answer, the exact filing chunks it was based on, and a breakdown of how long each pipeline stage took."),
    ]
    for num, color, title, detail in steps:
        st.markdown(f"""
<div style="display:flex;gap:14px;margin-bottom:10px;align-items:flex-start;">
  <div style="
    background:{color}20;border:1px solid {color}50;color:{color};
    width:24px;height:24px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:12px;font-weight:700;flex-shrink:0;margin-top:2px;
  ">{num}</div>
  <div>
    <span style="color:#e8eaf0;font-size:13px;font-weight:600">{title}</span><br>
    <span style="color:#6b7a99;font-size:12px;line-height:1.6">{detail}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
  <div style="margin-top:12px;padding:10px 14px;background:#060d18;border:1px solid #1a2235;border-radius:8px;">
    <span style="color:#4f8ef7;font-size:11px;font-weight:600;">💡 Why is this better than just asking ChatGPT?</span><br>
    <span style="color:#6b7a99;font-size:12px;">
      ChatGPT may have outdated data, cannot cite exact filing sections, and may hallucinate specific numbers.
      This system only uses the actual official SEC filings — every answer includes the exact source chunk so
      you can verify it yourself.
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── DATASET INFO PANEL ───────────────────────────────────────────────────────
with st.expander("📚  **About the Dataset** — What's indexed, what you can ask, how the pipeline works"):
    _render_dataset_info()

# ── CHAT HISTORY ─────────────────────────────────────────────────────────────
render_message_history(st.session_state.messages)

# Empty state when no messages yet
if not st.session_state.messages:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 48px 24px;
        color: #3d4f6b;
        border: 1px dashed #1a2235;
        border-radius: 14px;
        margin: 16px 0 24px;
    ">
        <div style="font-size:40px; margin-bottom:12px;">💬</div>
        <div style="font-size:16px; font-weight:600; color:#4f6080; margin-bottom:6px;">
            Start a conversation
        </div>
        <div style="font-size:13px; line-height:1.7;">
            Ask about revenue, risk factors, R&amp;D spend, business segments,<br>
            margins, or anything else in the 10-K filings.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── CHAT INPUT ───────────────────────────────────────────────────────────────
user_input: str | None = st.chat_input(
    "Ask about SEC filings… e.g. 'What was Apple's revenue in 2024?'"
)

if not user_input and st.session_state.get("pending_question"):
    user_input = st.session_state.pop("pending_question")

# ── PROCESS QUESTION ─────────────────────────────────────────────────────────
if user_input:
    user_msg = ChatMessage(role="user", content=user_input)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(user_input)

    assistant_msg = process_question(user_input, config, client)
    st.session_state.messages.append(assistant_msg)
