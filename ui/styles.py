"""
AI Nivid — UI Styling & Custom Dark CSS Theme.
"""

from __future__ import annotations

import streamlit as st

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #000000;
  --card: #0d0d0d;
  --border: #2a2a2a;
  --muted: #1a1a1a;
  --text: #ffffff;
  --muted-text: #b0b0b0;
  --primary: #ffffff;
  --green: #4ade80;
  --red: #f87171;
  --radius: 0.625rem;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stSidebar"] { display: none !important; }

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
  max-width: 1600px !important;
}

h1, h2, h3, h4 { color: var(--text) !important; font-weight: 700 !important; }
p, label, span, div { color: inherit; }
code, .stCode, [data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
}

/* Header */
.as-header {
  display: flex; align-items: center; gap: 1rem;
  padding: 1.25rem 0 1.5rem; margin-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
.as-logo {
  width: 48px; height: 48px; border-radius: 1rem;
  background: #fff; color: #000;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem; font-weight: 700;
}
.as-brand { font-size: 1.85rem; font-weight: 700; margin: 0; color: #fff; }
.as-sub { font-size: 0.875rem; color: #b0b0b0; margin: 0; }

.as-hero { text-align: center; padding: 2rem 1rem 2.5rem; }
.as-hero h2 {
  font-size: clamp(2rem, 4vw, 3rem); margin: 0 0 1rem; color: #fff !important;
}
.as-hero p {
  font-size: 1.15rem; color: #b0b0b0; max-width: 52rem;
  margin: 0 auto; line-height: 1.6;
}

/* Cards */
.as-card {
  background: rgba(13,13,13,0.85);
  border: 2px solid var(--border);
  border-radius: 1rem;
  padding: 1.75rem 1.5rem;
  margin-bottom: 1.5rem;
}
.as-card-title {
  display: flex; align-items: center; gap: 0.6rem;
  font-size: 1.4rem; font-weight: 700; margin: 0 0 0.35rem; color: #fff;
}
.as-card-desc { color: #b0b0b0; margin: 0 0 1.25rem; font-size: 1rem; }

.as-score-card {
  background: rgba(13,13,13,0.6);
  border: 2px solid var(--border);
  border-radius: 1rem;
  padding: 1.25rem;
  height: 100%;
}
.as-score-card.primary { border-color: #fff; }
.as-score-label { color: #b0b0b0; font-size: 1rem; margin-bottom: 0.5rem; }
.as-score-value {
  font-size: 2.75rem; font-weight: 700;
  font-family: 'IBM Plex Mono', monospace; color: #fff;
}
.as-score-value.sm { font-size: 2.1rem; }

.as-badge {
  display: inline-block; padding: 0.2rem 0.65rem;
  border-radius: 999px; font-size: 0.8rem; font-weight: 600;
  border: 1px solid var(--border); background: var(--muted); color: #fff;
}
.as-badge.good { background: #14532d; border-color: #22c55e; color: #86efac; }
.as-badge.warn { background: #1c1917; border-color: #a8a29e; color: #e7e5e4; }
.as-badge.bad { background: #333; border-color: #555; color: #fca5a5; }
.as-badge.outline {
  background: transparent; border-color: #666; color: #ddd;
}

.as-muted-box {
  background: rgba(26,26,26,0.55);
  border: 1px solid var(--border);
  border-radius: 1rem;
  padding: 1.25rem;
  margin: 0.75rem 0;
}
.as-primary-box {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 1rem;
  padding: 1.25rem;
  margin: 0.75rem 0;
}
.as-rec {
  display: flex; gap: 0.75rem; align-items: flex-start;
  background: rgba(26,26,26,0.55);
  border-left: 4px solid #fff;
  border-radius: 0.75rem;
  padding: 1rem;
  margin-bottom: 0.65rem;
  color: #e5e5e5;
}
.as-metric-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.55rem 0; border-bottom: 1px solid rgba(42,42,42,0.7);
  color: #cfcfcf;
}
.as-metric-row:last-child { border-bottom: none; }
.as-metric-val {
  font-family: 'IBM Plex Mono', monospace; font-size: 1.15rem; color: #fff;
}
.as-metric-val.good { color: var(--green); }
.as-metric-val.bad { color: var(--red); }

.as-footer {
  text-align: center; color: #888; padding: 2rem 0 1rem;
  margin-top: 2.5rem; border-top: 1px solid var(--border);
  font-size: 0.95rem;
}

/* Streamlit widgets */
.stButton > button {
  background: #fff !important; color: #000 !important;
  border: 1px solid #fff !important; border-radius: 0.5rem !important;
  font-weight: 600 !important; font-family: 'IBM Plex Sans', sans-serif !important;
  transition: opacity 0.15s ease;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button[kind="secondary"],
div[data-testid="stButton"] button[data-testid="baseButton-secondary"] {
  background: transparent !important; color: #fff !important;
  border: 1px solid #555 !important;
}

.stTextInput input, .stSelectbox > div > div, .stRadio > div {
  background: #111 !important; color: #fff !important;
  border-color: #333 !important;
}
.stTextInput input { border-radius: 0.5rem !important; }

[data-testid="stFileUploader"] {
  background: rgba(13,13,13,0.5);
  border: 2px dashed #2a2a2a !important;
  border-radius: 1rem !important;
  padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover {
  border-color: #888 !important;
  background: rgba(255,255,255,0.03);
}
[data-testid="stFileUploader"] section {
  border: none !important; background: transparent !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p { color: #999 !important; }

.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem; background: var(--muted);
  border-radius: 0.75rem; padding: 0.35rem;
}
.stTabs [data-baseweb="tab"] {
  color: #aaa !important; border-radius: 0.5rem !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  background: #222 !important; color: #fff !important;
}
.stProgress > div > div > div > div { background-color: #fff !important; }
.stAlert { border-radius: 0.75rem !important; }

hr { border-color: var(--border) !important; }
.stDataFrame, [data-testid="stDataFrame"] {
  border: 1px solid var(--border); border-radius: 0.75rem; overflow: hidden;
}

div[data-testid="stMetric"] {
  background: rgba(13,13,13,0.6);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 0.75rem 1rem;
}
div[data-testid="stMetric"] label { color: #aaa !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; }
</style>
"""


def apply_styles() -> None:
    """Inject custom dark theme styles into Streamlit page."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)
