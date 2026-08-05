"""
AI Shayak — Footer View.
"""

from __future__ import annotations

import streamlit as st


def render_footer() -> None:
    """Render application footer."""
    st.markdown(
        """
<div class="as-footer">
  © 2026 AI Shayak — Empowering Ethical Machine Learning
</div>
""",
        unsafe_allow_html=True,
    )
