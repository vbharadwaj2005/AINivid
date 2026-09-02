"""
AI Nivid — Core Ethical AI Governance & Audit Engine.
"""

from core.compliance import (
    build_report_text,
    generate_compliance_report,
    generate_mitigation_plan,
)
from core.config import (
    ALLOWED_MODEL_EXTS,
    MAX_DATASET_BYTES,
    MAX_MODEL_BYTES,
)
from core.evaluator import (
    _load_model_and_data,
    preview_dataset,
    run_deep_audit,
    run_evaluate,
    run_optimize,
    run_security_audit,
)
from core.fairness import (
    calculate_fairness_metrics,
    calculate_regression_fairness,
    calibration_parity,
    generate_retraining_suggestions,
    intersectional_analysis,
    tune_thresholds,
)
from core.models import (
    detect_model_framework,
    load_model,
    predict_model,
    predict_proba_model,
)
from core.security import (
    adversarial_robustness,
    audit_privacy,
    audit_robustness,
    differential_privacy_audit,
    membership_inference_attack,
)
from core.utils import (
    extract_target,
    is_numeric_dtype,
    logger,
    safe_run,
    temp_upload,
    validate_model_filename,
    validate_sensitive_column,
)

__all__ = [
    # Config
    "MAX_MODEL_BYTES",
    "MAX_DATASET_BYTES",
    "ALLOWED_MODEL_EXTS",
    # Utils
    "logger",
    "is_numeric_dtype",
    "temp_upload",
    "validate_sensitive_column",
    "validate_model_filename",
    "extract_target",
    "safe_run",
    # Models
    "detect_model_framework",
    "predict_model",
    "predict_proba_model",
    "load_model",
    # Fairness
    "calculate_fairness_metrics",
    "calculate_regression_fairness",
    "intersectional_analysis",
    "calibration_parity",
    "tune_thresholds",
    "generate_retraining_suggestions",
    # Security
    "adversarial_robustness",
    "differential_privacy_audit",
    "membership_inference_attack",
    "audit_robustness",
    "audit_privacy",
    # Compliance
    "generate_mitigation_plan",
    "generate_compliance_report",
    "build_report_text",
    # Evaluator
    "preview_dataset",
    "run_evaluate",
    "_load_model_and_data",
    "run_deep_audit",
    "run_optimize",
    "run_security_audit",
]
