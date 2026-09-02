"""
AI Nivid — Audit Summary Header & Score Cards View.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

import core
from ui.components import risk_badge


def render_summary(results: dict[str, Any]) -> None:
    """Render top summary header, report download button, and 4 top score cards."""
    r = results
    framework = r.get("framework")

    head_l, head_r = st.columns([3, 2])
    with head_l:
        st.markdown("## Audit Results")
    with head_r:
        b1, b2 = st.columns(2)
        with b1:
            if framework:
                st.markdown(
                    f'<div style="text-align:right;padding-top:0.6rem">'
                    f'<span class="as-badge outline">🧠 {framework}</span></div>',
                    unsafe_allow_html=True,
                )
        with b2:
            report = core.build_report_text(
                results=r,
                advanced=st.session_state.advanced,
                model_name=st.session_state.model_name or "N/A",
                dataset_name=st.session_state.dataset_name or "N/A",
                sensitive_column=st.session_state.get("sensitive_select")
                or st.session_state.get("sensitive_text")
                or "",
            )
            st.download_button(
                "Download Full Report",
                data=report,
                file_name=f"AI_Nivid_Audit_{datetime.now().date()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(
            f"""
            <div class="as-score-card primary">
              <div class="as-score-label">Overall Ethics Score</div>
              <div class="as-score-value">{r.get('ethicsScore', 0)}/10</div>
              <div style="margin-top:0.5rem">{risk_badge(r.get('riskLevel', 'Unknown'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sc2:
        fs = r.get("fairnessScore") or 0
        st.markdown(
            f"""
            <div class="as-score-card">
              <div class="as-score-label">Fairness</div>
              <div class="as-score-value sm">{fs}/10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(fs / 10, 0), 1.0))
    with sc3:
        ints = r.get("integrityScore") or 0
        st.markdown(
            f"""
            <div class="as-score-card">
              <div class="as-score-label">Data Integrity</div>
              <div class="as-score-value sm">{ints}/10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(ints / 10, 0), 1.0))
    with sc4:
        ts = r.get("transparencyScore") or 0
        st.markdown(
            f"""
            <div class="as-score-card">
              <div class="as-score-label">Transparency</div>
              <div class="as-score-value sm">{ts}/10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(ts / 10, 0), 1.0))

    st.markdown("<br/>", unsafe_allow_html=True)
