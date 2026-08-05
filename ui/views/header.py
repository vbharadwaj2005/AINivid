"""
AI Shayak — Header and Hero Banner View.
"""

from __future__ import annotations

import streamlit as st


def render_header_hero() -> None:
    """Render top header brand bar and hero headline."""
    st.markdown(
        """
<div class="as-header">
  <div class="as-logo">✓</div>
  <div>
    <p class="as-brand">AI Shayak</p>
    <p class="as-sub">Ethical AI Governance Platform</p>
  </div>
</div>
<div class="as-hero">
  <h2>AI Ethics &amp; Bias Audit</h2>
  <p>Multi-framework fairness auditing, security analysis, threshold optimization,
  deep fairness metrics, and regulatory compliance for ML models.</p>
</div>
""",
        unsafe_allow_html=True,
    )
