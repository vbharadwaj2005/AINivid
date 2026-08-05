"""
AI Shayak — Configuration and Constants.
"""

from __future__ import annotations

# File size limits (500 MB)
MAX_MODEL_BYTES: int = 500 * 1024 * 1024
MAX_DATASET_BYTES: int = 500 * 1024 * 1024

# Supported model file extensions
ALLOWED_MODEL_EXTS: set[str] = {
    ".pkl",
    ".joblib",
    ".json",
    ".ubj",
    ".pt",
    ".pth",
    ".h5",
    ".keras",
}
