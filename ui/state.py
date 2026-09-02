"""
AI Nivid — Session State Management Helpers.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import streamlit as st


def init_state() -> None:
    """Initialize default Streamlit session state keys."""
    defaults: dict[str, Any] = {
        "results": None,
        "advanced": {},
        "schema": None,
        "show_schema": False,
        "error": "",
        "column_names": [],
        "eval_type": "model",
        "model_bytes": None,
        "model_name": None,
        "dataset_bytes": None,
        "dataset_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def store_upload(key_bytes: str, key_name: str, uploaded: Any) -> None:
    """Store uploaded file bytes and filename in session state."""
    if uploaded is not None:
        st.session_state[key_bytes] = uploaded.getvalue()
        st.session_state[key_name] = uploaded.name


def file_buf(key_bytes: str) -> Optional[io.BytesIO]:
    """Retrieve BytesIO buffer for uploaded file data stored in session state."""
    data = st.session_state.get(key_bytes)
    if data is None:
        return None
    return io.BytesIO(data)
