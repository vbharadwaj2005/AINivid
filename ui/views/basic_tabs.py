"""
AI Shayak — Primary Audit Result Tabs View.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ui.components import status_badge


def render_basic_tabs(results: dict[str, Any]) -> None:
    """Render the primary evaluation tabs (Bias, Security, Group Analysis, Compliance, Recommendations)."""
    r = results
    metrics = r.get("metrics") or {}
    compliance = r.get("compliance") or {}

    tab_bias, tab_sec, tab_grp, tab_comp, tab_rec = st.tabs(
        [
            "⚠ Bias & Fairness",
            "🛡 Security Audit",
            "👥 Group Analysis",
            "📄 Compliance",
            "✓ Recommendations",
        ]
    )

    with tab_bias:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="as-muted-box">
                  <h4 style="margin-top:0;color:#fff">Parity Metrics</h4>
                  <div class="as-metric-row">
                    <span>Demographic Parity Diff</span>
                    <span class="as-metric-val">{metrics.get('demographicParityDifference', 0)}</span>
                  </div>
                  <div class="as-metric-row">
                    <span>Equalized Odds Diff</span>
                    <span class="as-metric-val">{metrics.get('equalizedOddsDifference', 0)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            di = metrics.get("disparateImpact") or 0
            di_cls = "bad" if di < 0.8 else "good"
            st.markdown(
                f"""
                <div class="as-muted-box">
                  <h4 style="margin-top:0;color:#fff">Impact Ratios</h4>
                  <div class="as-metric-row">
                    <span>Disparate Impact Ratio</span>
                    <span class="as-metric-val {di_cls}">{di}</span>
                  </div>
                  <div class="as-metric-row">
                    <span>Mitigation Strategy</span>
                    <span class="as-badge outline">Post-Processing</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        rec0 = (r.get("recommendations") or ["No critical bias detected."])[0]
        st.markdown(
            f"""
            <div class="as-primary-box">
              <h4 style="margin-top:0;color:#fff">🧠 Mitigation Plan</h4>
              <p style="color:#cfcfcf;font-size:1.05rem;margin:0">{rec0}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_sec:
        s1, s2 = st.columns(2)
        rob = metrics.get("robustness") or 0
        priv = metrics.get("privacy") or 0
        with s1:
            st.markdown(
                f"""
                <div class="as-score-card">
                  <div class="as-score-label">🛡 Robustness</div>
                  <div class="as-score-value sm">{rob * 100:.1f}%</div>
                  <p style="color:#888;font-size:0.85rem">Stability under ±5% Gaussian noise.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(rob, 0), 1.0))
        with s2:
            st.markdown(
                f"""
                <div class="as-score-card">
                  <div class="as-score-label">🔍 Privacy Leakage</div>
                  <div class="as-score-value sm">{priv * 100:.1f}%</div>
                  <p style="color:#888;font-size:0.85rem">Privacy score (higher = less leakage).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(priv, 0), 1.0))

    with tab_grp:
        g1, g2 = st.columns(2)
        chart_data = r.get("chartData") or []
        eval_mode = st.session_state.eval_type
        with g1:
            title = (
                "Selection Rate by Group"
                if eval_mode == "model"
                else "Dataset Distribution"
            )
            st.markdown(f"**{title}**")
            if chart_data:
                cdf = pd.DataFrame(chart_data)
                st.bar_chart(
                    cdf.set_index("group")["value"],
                    color="#ffffff",
                    use_container_width=True,
                )
        with g2:
            gp = r.get("groupPerformance") or {}
            rows = []
            for g, d in gp.items():
                if eval_mode == "model":
                    rows.append(
                        {
                            "Group": g,
                            "Selection Rate": (
                                f"{d.get('selectionRate', 0):.3f}"
                                if d.get("selectionRate") is not None
                                else "N/A"
                            ),
                            "Accuracy": (
                                f"{(d.get('accuracy') or 0) * 100:.1f}%"
                            ),
                            "Count": d.get("count"),
                        }
                    )
                else:
                    rows.append(
                        {
                            "Group": g,
                            "Count": d.get("count"),
                            "Proportion": f"{(d.get('percentage') or 0) * 100:.1f}%",
                        }
                    )
            if rows:
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )

    with tab_comp:
        fws = compliance.get("frameworks") or []
        if fws:
            fc1, fc2 = st.columns(2)
            for i, fw in enumerate(fws):
                target = fc1 if i % 2 == 0 else fc2
                with target:
                    st.markdown(
                        f"""
                        <div class="as-muted-box">
                          <div style="display:flex;justify-content:space-between;align-items:start">
                            <h4 style="margin:0;color:#fff">{fw.get('name','')}</h4>
                            {status_badge(fw.get('status',''))}
                          </div>
                          <p style="color:#fff;font-weight:600;font-size:0.9rem;margin:0.75rem 0 0.35rem">
                            {fw.get('requirement','')}
                          </p>
                          <p style="color:#999;margin:0">{fw.get('details','')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        card = compliance.get("modelCard") or {}
        if any(card.values()):
            st.markdown("#### Generated Model Card")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("**INTENDED USE**")
                st.write(card.get("intendedUse") or "—")
            with m2:
                st.markdown("**FAIRNESS PHILOSOPHY**")
                st.write(card.get("fairnessPhilosophy") or "—")
            with m3:
                st.markdown("**LIMITATIONS**")
                st.write(card.get("limitations") or "—")

    with tab_rec:
        for rec in r.get("recommendations") or []:
            st.markdown(
                f'<div class="as-rec"><span>✓</span><span>{rec}</span></div>',
                unsafe_allow_html=True,
            )
