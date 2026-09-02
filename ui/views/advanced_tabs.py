"""
AI Nivid — Advanced Analysis Tabs View (Optimization, Deep Fairness, Security).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import core
from ui.components import status_badge
from ui.state import file_buf


def render_advanced_tabs() -> None:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="as-card">
          <div class="as-card-title">⚙ Advanced Analysis</div>
          <p class="as-card-desc">Threshold optimization, retraining suggestions,
          deep fairness metrics, and security vulnerability assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sens_col = (
        st.session_state.get("sensitive_select")
        or st.session_state.get("sensitive_text")
        or ""
    )

    target_col = (
        st.session_state.get("target_select")
        or st.session_state.get("target_text")
        or None
    )

    def _require_model_assets() -> bool:
        if not (
            st.session_state.model_bytes
            and st.session_state.dataset_bytes
            and sens_col.strip()
        ):
            st.warning(
                "Upload model, dataset, and specify sensitive column first "
                "(End-to-End Audit required)."
            )
            return False
        return True

    adv_opt, adv_deep, adv_sec = st.tabs(
        ["🎚 Optimization", "▦ Deep Fairness", "🔒 Security"]
    )

    with adv_opt:
        o1, o2, o3 = st.columns(3)
        with o1:
            if st.button("Tune Thresholds", use_container_width=True, key="btn_thr"):
                if _require_model_assets():
                    with st.spinner("Tuning thresholds..."):
                        st.session_state.advanced["threshold"] = core.safe_run(
                            core.run_optimize,
                            file_buf("model_bytes"),
                            st.session_state.model_name,
                            file_buf("dataset_bytes"),
                            st.session_state.dataset_name,
                            sens_col,
                            "threshold",
                            target_col,
                        )
        with o2:
            if st.button(
                "Retraining Suggestions", use_container_width=True, key="btn_retrain"
            ):
                if _require_model_assets():
                    with st.spinner("Analyzing feature correlations..."):
                        st.session_state.advanced["retraining"] = core.safe_run(
                            core.run_optimize,
                            file_buf("model_bytes"),
                            st.session_state.model_name,
                            file_buf("dataset_bytes"),
                            st.session_state.dataset_name,
                            sens_col,
                            "retraining",
                            target_col,
                        )
        with o3:
            if st.button(
                "Regression Fairness", use_container_width=True, key="btn_reg"
            ):
                if _require_model_assets():
                    with st.spinner("Computing regression fairness..."):
                        st.session_state.advanced["regression-fairness"] = (
                            core.safe_run(
                                core.run_optimize,
                                file_buf("model_bytes"),
                                st.session_state.model_name,
                                file_buf("dataset_bytes"),
                                st.session_state.dataset_name,
                                sens_col,
                                "regression-fairness",
                                target_col,
                            )
                        )

        thr = st.session_state.advanced.get("threshold")
        if thr:
            if thr.get("error"):
                st.error(thr["error"])
            else:
                t1, t2, t3 = st.columns(3)
                with t1:
                    of = thr.get("optimalFairness") or {}
                    st.metric(
                        "Optimal Fairness",
                        of.get("threshold"),
                        f"SPD: {of.get('spd')}",
                    )
                with t2:
                    oa = thr.get("optimalAccuracy") or {}
                    st.metric(
                        "Best Accuracy",
                        oa.get("threshold"),
                        f"Acc: {(oa.get('accuracy') or 0) * 100:.1f}%",
                    )
                with t3:
                    rec = thr.get("recommended") or {}
                    st.metric(
                        "Recommended",
                        rec.get("threshold"),
                        f"SPD: {rec.get('spd')} Acc: {(rec.get('accuracy') or 0) * 100:.1f}%",
                    )
                st.info(thr.get("recommendation", ""))
                rows = []
                for t in thr.get("thresholds") or []:
                    rows.append(
                        {
                            "Threshold": t.get("threshold"),
                            "SPD": t.get("spd"),
                            "DI": t.get("di"),
                            "Accuracy": f"{(t.get('accuracy') or 0) * 100:.1f}%",
                            "Bias": "Bias" if t.get("biasDetected") else "Fair",
                        }
                    )
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows), use_container_width=True, hide_index=True
                    )

        retr = st.session_state.advanced.get("retraining")
        if retr:
            if retr.get("error"):
                st.error(retr["error"])
            else:
                biased = retr.get("potentialBiasedFeatures") or []
                if biased:
                    st.warning("Biased Features: " + ", ".join(biased))
                st.markdown(
                    f"**Resampling:** {retr.get('resamplingSuggestion', '')}"
                )
                st.markdown(
                    f"**Reweighting:** {retr.get('reweightingSuggestion', '')}"
                )
                if retr.get("adversarialDebiasingSuggestion"):
                    st.markdown(
                        f"**Advanced:** {retr['adversarialDebiasingSuggestion']}"
                    )
                corrs = retr.get("featureCorrelations") or []
                if corrs:
                    st.dataframe(
                        pd.DataFrame(corrs).rename(
                            columns={
                                "feature": "Feature",
                                "correlation_with_sensitive": "Correlation w/ Sensitive",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        reg = st.session_state.advanced.get("regression-fairness")
        if reg:
            if reg.get("error"):
                st.error(reg["error"])
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric(
                    "Mean Prediction Diff", reg.get("meanPredictionDifference")
                )
                r2.metric("MAE Disparity", reg.get("maxMaeDisparity"))
                r3.metric("Overall Mean", reg.get("overallMean"))

    with adv_deep:
        deep_type = st.radio(
            "Deep audit type",
            ["intersectional", "calibration"],
            format_func=lambda x: (
                "Intersectional" if x == "intersectional" else "Calibration"
            ),
            horizontal=True,
            key="deep_type",
            label_visibility="collapsed",
        )
        second_col = ""
        if deep_type == "intersectional":
            opts = [
                c
                for c in (st.session_state.column_names or [])
                if c != sens_col
            ]
            if opts:
                second_col = st.selectbox(
                    "2nd Column", options=opts, key="second_sensitive"
                )
            else:
                second_col = st.text_input(
                    "2nd Column", placeholder="e.g., race", key="second_sensitive_txt"
                )

        if st.button(f"Run {deep_type}", type="primary", key="btn_deep"):
            if _require_model_assets():
                with st.spinner(f"Running {deep_type} analysis..."):
                    cols = [sens_col]
                    if deep_type == "intersectional":
                        cols.append(second_col or sens_col)
                    st.session_state.advanced[deep_type] = core.safe_run(
                        core.run_deep_audit,
                        file_buf("model_bytes"),
                        st.session_state.model_name,
                        file_buf("dataset_bytes"),
                        st.session_state.dataset_name,
                        cols,
                        deep_type,
                        target_col,
                    )

        inter = st.session_state.advanced.get("intersectional")
        if inter:
            if inter.get("error"):
                st.error(inter["error"])
            else:
                fair = inter.get("fairness") or {}
                i1, i2, i3, i4 = st.columns(4)
                i1.metric("SPD", round(fair.get("spd", 0), 4))
                i2.metric("DI", round(fair.get("di", 0), 4))
                i3.metric("EOD", round(fair.get("eod", 0), 4))
                i4.metric("AOD", round(fair.get("aod", 0), 4))
                cd = inter.get("chartData") or []
                if cd:
                    st.bar_chart(
                        pd.DataFrame(cd).set_index("group")["value"],
                        color="#ffffff",
                        use_container_width=True,
                    )

        cal = st.session_state.advanced.get("calibration")
        if cal:
            if cal.get("error"):
                st.error(cal["error"])
            else:
                st.metric(
                    "Calibration Error",
                    cal.get("calibrationError"),
                    help="Lower is better",
                )
                cal_data = cal.get("calibrationData") or {}
                for group, bins in cal_data.items():
                    st.markdown(f"**{group}**")
                    rows = []
                    for b, v in (bins or {}).items():
                        rows.append(
                            {
                                "Bin": b,
                                "Mean Pred": round(v.get("meanPred", 0), 3)
                                if isinstance(v.get("meanPred"), (int, float))
                                else v.get("meanPred"),
                                "Mean Actual": round(v.get("meanActual", 0), 3)
                                if isinstance(v.get("meanActual"), (int, float))
                                else v.get("meanActual"),
                                "Count": v.get("count"),
                            }
                        )
                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )

    with adv_sec:
        s1, s2, s3 = st.columns(3)
        with s1:
            if st.button(
                "Adversarial Robustness", use_container_width=True, key="btn_adv"
            ):
                if _require_model_assets():
                    with st.spinner("Running adversarial attack..."):
                        st.session_state.advanced["adversarial"] = core.safe_run(
                            core.run_security_audit,
                            file_buf("model_bytes"),
                            st.session_state.model_name,
                            file_buf("dataset_bytes"),
                            st.session_state.dataset_name,
                            sens_col,
                            "adversarial",
                            target_col,
                        )
        with s2:
            if st.button(
                "Differential Privacy", use_container_width=True, key="btn_dp"
            ):
                if _require_model_assets():
                    with st.spinner("Estimating DP epsilon..."):
                        st.session_state.advanced["differential-privacy"] = (
                            core.safe_run(
                                core.run_security_audit,
                                file_buf("model_bytes"),
                                st.session_state.model_name,
                                file_buf("dataset_bytes"),
                                st.session_state.dataset_name,
                                sens_col,
                                "differential-privacy",
                                target_col,
                            )
                        )
        with s3:
            if st.button(
                "Membership Inference", use_container_width=True, key="btn_mi"
            ):
                if _require_model_assets():
                    with st.spinner("Running membership inference..."):
                        st.session_state.advanced["membership-inference"] = (
                            core.safe_run(
                                core.run_security_audit,
                                file_buf("model_bytes"),
                                st.session_state.model_name,
                                file_buf("dataset_bytes"),
                                st.session_state.dataset_name,
                                sens_col,
                                "membership-inference",
                                target_col,
                            )
                        )

        adv = st.session_state.advanced.get("adversarial")
        if adv:
            if adv.get("error"):
                st.error(adv["error"])
            else:
                a1, a2, a3 = st.columns(3)
                a1.metric(
                    "Accuracy Under Attack",
                    f"{(adv.get('accuracyUnderAttack') or 0) * 100:.1f}%",
                )
                a2.metric(
                    "Attack Success Rate",
                    f"{(adv.get('attackSuccessRate') or 0) * 100:.1f}%",
                )
                a3.metric(
                    "Perturbation ε",
                    adv.get("perturbationEpsilon"),
                    adv.get("perturbationDescription"),
                )

        dp = st.session_state.advanced.get("differential-privacy")
        if dp:
            if dp.get("error"):
                st.error(dp["error"])
            else:
                d1, d2, d3 = st.columns(3)
                d1.metric("DP Epsilon", dp.get("estimatedEpsilon"))
                d2.metric("Max Influence", dp.get("maxInfluence"))
                with d3:
                    st.markdown("**Risk Level**")
                    st.markdown(
                        status_badge(dp.get("riskLevel", "")),
                        unsafe_allow_html=True,
                    )
                st.caption(dp.get("interpretation", ""))

        mi = st.session_state.advanced.get("membership-inference")
        if mi:
            if mi.get("error"):
                st.error(mi["error"])
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Member Confidence",
                    f"{(mi.get('memberConfidence') or 0) * 100:.1f}%",
                )
                m2.metric(
                    "Non-Member Confidence",
                    f"{(mi.get('nonMemberConfidence') or 0) * 100:.1f}%",
                )
                m3.metric("Risk Score", mi.get("riskScore"))
                with m4:
                    st.markdown("**Risk Level**")
                    st.markdown(
                        status_badge(mi.get("riskLevel", "")),
                        unsafe_allow_html=True,
                    )
                st.caption(mi.get("interpretation", ""))
