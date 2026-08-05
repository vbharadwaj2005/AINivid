"""
AI Shayak — Utility and Helper Functions.
"""

from __future__ import annotations

import logging
import os
import tempfile
import traceback
from contextlib import contextmanager
from typing import Any, BinaryIO, Callable, Iterator, Optional

import numpy as np
import pandas as pd

from core.config import ALLOWED_MODEL_EXTS

logger = logging.getLogger("aishayak.core")


def is_numeric_dtype(series_or_dtype: Any) -> bool:
    try:
        return bool(pd.api.types.is_numeric_dtype(series_or_dtype))
    except Exception:
        return False


@contextmanager
def temp_upload(
    uploaded_file: BinaryIO, filename: str, max_bytes: int
) -> Iterator[str]:
    ext = os.path.splitext(filename)[1].lower()
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        os.close(fd)
        data = uploaded_file.read()
        if len(data) > max_bytes:
            raise ValueError(
                f"File exceeds maximum size ({max_bytes // (1024 * 1024)}MB)"
            )
        with open(path, "wb") as f:
            f.write(data)
        yield path
    finally:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError as e:
            logger.warning("Failed to clean up temp file %s: %s", path, e)


def validate_sensitive_column(df: pd.DataFrame, column: str) -> str:
    column = (column or "").strip()
    if not column:
        raise ValueError("Sensitive column name is required")
    if column not in df.columns:
        raise ValueError(
            f'Sensitive column "{column}" not found in dataset columns: {list(df.columns)}'
        )
    return column


def validate_model_filename(filename: str) -> str:
    if not filename:
        raise ValueError("Model filename is required")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_MODEL_EXTS:
        raise ValueError(
            f"Unsupported model format: {ext}. Allowed formats: {', '.join(sorted(ALLOWED_MODEL_EXTS))}"
        )
    return filename


def extract_target(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> tuple[pd.DataFrame, pd.Series | None, str]:
    """Extract feature set X and target y for any general AI model or dataset."""
    selected_col = None

    if target_col and target_col in df.columns:
        selected_col = target_col
    else:
        candidates = ["target", "label", "class", "y", "income", "output", "response", "outcome"]
        for c in candidates:
            if c in df.columns:
                selected_col = c
                break
        if not selected_col and len(df.columns) > 1:
            selected_col = df.columns[-1]

    if not selected_col or selected_col not in df.columns:
        return df, None, ""

    X = df.drop(selected_col, axis=1)
    raw_y = df[selected_col]

    if is_numeric_dtype(raw_y):
        return X, raw_y, selected_col

    str_y = raw_y.astype(str).str.strip()
    unique_vals = list(str_y.unique())

    if len(unique_vals) == 2:
        pos_candidates = {">50k", "1", "1.0", "true", "yes", "positive", "pass", "approved", "high"}
        u0_lower = unique_vals[0].lower()
        if u0_lower in pos_candidates:
            pos_val = unique_vals[0]
        else:
            pos_val = unique_vals[1]
        y_binary = (str_y == pos_val).astype(int)
        return X, y_binary, selected_col

    codes, _ = pd.factorize(str_y)
    return X, pd.Series(codes, index=df.index), selected_col


def safe_run(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.error("%s failed: %s", fn.__name__, traceback.format_exc())
        return {"error": str(e)}
