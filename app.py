"""
AI Shayak — Ethical AI Governance Platform (Streamlit)
Main entry point for running the Streamlit application.
"""

from __future__ import annotations

import streamlit as st

from ui import render_app

st.set_page_config(
    page_title="AI Shayak — Ethical AI Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_app()
