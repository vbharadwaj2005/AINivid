"""
AI Shayak — User Interface Package.
"""

from __future__ import annotations

import logging

import streamlit as st

from ui.state import init_state
from ui.styles import apply_styles
from ui.views.advanced_tabs import render_advanced_tabs
from ui.views.basic_tabs import render_basic_tabs
from ui.views.footer import render_footer
from ui.views.header import render_header_hero
from ui.views.inputs import render_input_assets
from ui.views.summary import render_summary

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def render_app() -> None:
    """Render the full Streamlit AI Shayak application UI."""
    apply_styles()
    init_state()
    render_header_hero()
    render_input_assets()

    results = st.session_state.results
    if results:
        render_summary(results)
        render_basic_tabs(results)
        render_advanced_tabs()

    render_footer()


__all__ = ["render_app"]
