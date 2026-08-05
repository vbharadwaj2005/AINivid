"""
AI Shayak — Model Framework Detection, Loading, and Prediction Wrappers.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import numpy as np

from core.config import ALLOWED_MODEL_EXTS


def detect_model_framework(model: Any) -> str:
    """Detect framework for any general AI model architecture."""
    module_name = getattr(type(model), "__module__", "").lower()
    if "sklearn" in module_name:
        return "sklearn"
    if "xgboost" in module_name:
        return "xgboost"
    if "lightgbm" in module_name:
        return "lightgbm"
    if "catboost" in module_name:
        return "catboost"
    if "torch" in module_name or "pytorch" in module_name:
        return "pytorch"
    if "tensorflow" in module_name or "keras" in module_name:
        return "tensorflow"
    if "transformers" in module_name or "huggingface" in module_name:
        return "transformers"
    if "onnx" in module_name:
        return "onnx"
    if callable(model):
        return "callable"
    return "general"


def predict_model(model: Any, X: Any, framework: Optional[str] = None) -> Any:
    """Unified prediction interface supporting classification, regression, and general AI models."""
    if framework is None:
        framework = detect_model_framework(model)

    if framework in ("sklearn", "xgboost", "lightgbm", "catboost"):
        return model.predict(X)

    if framework == "pytorch":
        import torch

        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            values = X.values if hasattr(X, "values") else X
            X_tensor = torch.tensor(values, dtype=torch.float32)
            out = model(X_tensor)
            if hasattr(out, "logits"):
                out = out.logits
            if out.dim() > 1 and out.shape[1] > 1:
                return out.argmax(dim=1).cpu().numpy()
            return (out.squeeze() > 0.5).cpu().numpy().astype(int)

    if framework == "tensorflow":
        out = model.predict(X, verbose=0)
        if hasattr(out, "ndim") and out.ndim > 1 and out.shape[1] > 1:
            return out.argmax(axis=1)
        return (np.squeeze(out) > 0.5).astype(int)

    if framework == "transformers":
        if hasattr(model, "predict"):
            return model.predict(X)
        out = model(X)
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            return np.array([item.get("label", item.get("score", 0)) for item in out])
        return np.array(out)

    if hasattr(model, "predict"):
        return model.predict(X)

    if callable(model):
        return model(X)

    raise ValueError(f"Unable to execute predictions for framework/model type: {type(model)}")


def predict_proba_model(model: Any, X: Any, framework: Optional[str] = None) -> Any:
    """Unified prediction probability interface for model benchmark evaluations."""
    if framework is None:
        framework = detect_model_framework(model)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if framework in ("xgboost", "lightgbm", "catboost"):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return model.predict(X)

    if framework == "pytorch":
        import torch

        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            values = X.values if hasattr(X, "values") else X
            X_tensor = torch.tensor(values, dtype=torch.float32)
            raw = model(X_tensor)
            if hasattr(raw, "logits"):
                raw = raw.logits
            out = torch.softmax(raw, dim=1) if raw.dim() > 1 and raw.shape[1] > 1 else torch.sigmoid(raw)
            return out.cpu().numpy()

    if framework == "tensorflow":
        out = model.predict(X, verbose=0)
        return out

    if hasattr(model, "decision_function"):
        df_vals = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-df_vals))

    preds = predict_model(model, X, framework)
    return preds


def load_model(model_path: str, filename: str) -> tuple[Any, str]:
    """Load model artifact from file path."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_MODEL_EXTS:
        raise ValueError(f"Unsupported model format: {ext}")

    if ext in (".pkl", ".joblib"):
        model = joblib.load(model_path)
        return model, detect_model_framework(model)

    if ext in (".json", ".ubj"):
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            return model, "xgboost"
        except Exception:
            model = joblib.load(model_path)
            return model, detect_model_framework(model)

    if ext in (".pt", ".pth"):
        import torch
        try:
            model = torch.jit.load(model_path)
            model.eval()
            return model, "pytorch"
        except Exception:
            model = torch.load(model_path, map_location="cpu")
            return model, "pytorch"

    if ext in (".h5", ".keras", ".pb"):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, "tensorflow"

    raise ValueError(f"Unsupported model format: {ext}")
