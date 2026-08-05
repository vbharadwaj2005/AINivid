"""
AI Shayak — Input Assets and Schema Preview View.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import core
from ui.state import file_buf, store_upload


def render_input_assets() -> None:
    st.markdown(
        """
<div class="as-card">
  <div class="as-card-title">⬆ Input Assets</div>
  <p class="as-card-desc">Upload ML artifacts for evaluation. Supports scikit-learn,
  XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow, Transformers, and Custom Callables.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    eval_type = st.radio(
        "Audit Scope",
        options=["model", "dataset"],
        format_func=lambda x: (
            "End-to-End Audit" if x == "model" else "Data-Only Ethics Audit"
        ),
        horizontal=True,
        key="eval_type_radio",
    )
    st.session_state.eval_type = eval_type

    col_model, col_data = st.columns(2)
    with col_model:
        st.markdown("**Model Artifact**")
        st.caption(".pkl · .joblib · .json · .ubj · .pt · .pth · .h5 · .keras")
        model_up = st.file_uploader(
            "Drop or click to upload model",
            type=["pkl", "joblib", "json", "ubj", "pt", "pth", "h5", "keras"],
            key="model_uploader",
            disabled=(eval_type == "dataset"),
            label_visibility="collapsed",
        )
        if model_up and eval_type != "dataset":
            if model_up.size > core.MAX_MODEL_BYTES:
                st.error("Model file exceeds maximum size (500MB)")
            else:
                store_upload("model_bytes", "model_name", model_up)
                st.markdown(
                    f'<span class="as-badge outline">{model_up.name}</span>',
                    unsafe_allow_html=True,
                )

    with col_data:
        st.markdown("**Dataset Samples**")
        st.caption("CSV format with target column and features")
        data_up = st.file_uploader(
            "Drop or click to upload dataset",
            type=["csv"],
            key="dataset_uploader",
            label_visibility="collapsed",
        )
        if data_up:
            if data_up.size > core.MAX_DATASET_BYTES:
                st.error("Dataset file exceeds maximum size (500MB)")
            else:
                store_upload("dataset_bytes", "dataset_name", data_up)
                st.markdown(
                    f'<span class="as-badge outline">{data_up.name}</span>',
                    unsafe_allow_html=True,
                )
                try:
                    preview_head = pd.read_csv(
                        io.BytesIO(st.session_state.dataset_bytes), nrows=0
                    )
                    st.session_state.column_names = list(preview_head.columns)
                except Exception:
                    pass

    has_dataset = st.session_state.dataset_bytes is not None
    has_model = st.session_state.model_bytes is not None

    if has_dataset:
        schema_cols = st.columns([1, 4])
        with schema_cols[0]:
            if st.button("Preview Schema", use_container_width=True):
                with st.spinner("Loading schema..."):
                    result = core.safe_run(
                        core.preview_dataset,
                        file_buf("dataset_bytes"),
                        st.session_state.dataset_name or "data.csv",
                    )
                    if result.get("error"):
                        st.session_state.error = result["error"]
                    else:
                        st.session_state.schema = result
                        st.session_state.show_schema = True
                        st.session_state.column_names = result.get(
                            "columnNames", []
                        )
                        st.session_state.error = ""

    if st.session_state.show_schema and st.session_state.schema:
        sch = st.session_state.schema
        st.markdown(
            f"""
            <div class="as-muted-box">
              <strong>Dataset Schema</strong> —
              {sch.get('rowCount', 0)} rows × {sch.get('columnCount', 0)} columns
            </div>
            """,
            unsafe_allow_html=True,
        )
        rows = []
        for c in sch.get("columns") or []:
            samples = (
                ", ".join(str(x) for x in (c.get("sampleValues") or [])[:3]) or "-"
            )
            rows.append(
                {
                    "Column": c.get("name"),
                    "Type": c.get("dtype"),
                    "Unique": c.get("uniqueCount"),
                    "Missing": c.get("missingCount"),
                    "Sample": samples,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if has_dataset or (has_model and eval_type == "model"):
        st.markdown('<div class="as-muted-box">', unsafe_allow_html=True)
        c_sens, c_target = st.columns(2)

        suggestions = st.session_state.column_names or []

        with c_sens:
            st.markdown("**Sensitive Attribute**")
            if suggestions:
                default_idx = 0
                for preferred in ("sex", "race", "gender", "age", "ethnicity", "region"):
                    if preferred in suggestions:
                        default_idx = suggestions.index(preferred)
                        break
                sensitive = st.selectbox(
                    "Protected demographic group column for fairness analysis.",
                    options=suggestions,
                    index=default_idx,
                    key="sensitive_select",
                    label_visibility="collapsed",
                )
            else:
                sensitive = st.text_input(
                    "Protected demographic group column",
                    placeholder="e.g., race, gender, age",
                    key="sensitive_text",
                    label_visibility="collapsed",
                )
            st.caption("Protected demographic group column for fairness analysis.")

        with c_target:
            st.markdown("**Target Outcome Column**")
            if suggestions:
                target_default_idx = len(suggestions) - 1
                candidates = ["target", "label", "class", "y", "income", "output", "response", "outcome"]
                for cand in candidates:
                    if cand in suggestions:
                        target_default_idx = suggestions.index(cand)
                        break
                target_col = st.selectbox(
                    "Target label column for prediction comparison.",
                    options=suggestions,
                    index=target_default_idx,
                    key="target_select",
                    label_visibility="collapsed",
                )
            else:
                target_col = st.text_input(
                    "Target label column",
                    placeholder="e.g., target, label, class",
                    key="target_text",
                    label_visibility="collapsed",
                )
            st.caption("Target label/outcome column in the dataset.")

        run_clicked = st.button(
            "Run Ethics Evaluation",
            type="primary",
            use_container_width=True,
            key="run_eval",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if run_clicked:
            st.session_state.advanced = {}
            st.session_state.error = ""
            need_model = eval_type == "model"
            target_val = (target_col or "").strip()
            if need_model and not has_model:
                st.session_state.error = (
                    "Upload model, dataset, and specify sensitive column"
                )
            elif not has_dataset or not (sensitive or "").strip():
                st.session_state.error = (
                    "Upload dataset and specify sensitive column"
                    if not need_model
                    else "Upload model, dataset, and specify sensitive column"
                )
            else:
                with st.spinner("Running ethics audit..."):
                    progress = st.progress(20, text="Preparing assets...")
                    try:
                        progress.progress(
                            40, text="Evaluating fairness & integrity..."
                        )
                        result = core.run_evaluate(
                            dataset_file=file_buf("dataset_bytes"),
                            dataset_name=st.session_state.dataset_name or "data.csv",
                            sensitive_column=sensitive.strip(),
                            target_column=target_val if target_val else None,
                            evaluation_type=eval_type,
                            model_file=file_buf("model_bytes") if need_model else None,
                            model_name=st.session_state.model_name
                            if need_model
                            else None,
                        )
                        progress.progress(100, text="Complete")
                        st.session_state.results = result
                        st.session_state.error = ""
                    except Exception as e:
                        st.session_state.results = None
                        st.session_state.error = str(e)
                    finally:
                        progress.empty()

    if st.session_state.error:
        st.error(f"**Audit Failed** — {st.session_state.error}")
