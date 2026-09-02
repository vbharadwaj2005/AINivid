"""
AI Nivid — High-level Audit Evaluation & Orchestration APIs.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Optional

import pandas as pd

from core.compliance import (
    generate_compliance_report,
    generate_mitigation_plan,
)
from core.config import MAX_DATASET_BYTES, MAX_MODEL_BYTES
from core.fairness import (
    calculate_fairness_metrics,
    calculate_regression_fairness,
    calibration_parity,
    generate_retraining_suggestions,
    intersectional_analysis,
    tune_thresholds,
)
from core.models import load_model, predict_model, predict_proba_model
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
    temp_upload,
    validate_model_filename,
    validate_sensitive_column,
)


def preview_dataset(dataset_file: BinaryIO, filename: str = "data.csv") -> dict[str, Any]:
    """Inspect CSV dataset, returning metadata, row/column counts, and sample values."""
    with temp_upload(dataset_file, filename, MAX_DATASET_BYTES) as path:
        df = pd.read_csv(path)

    columns = []
    detected_target = None
    candidates = ["target", "label", "class", "y", "income", "output", "response", "outcome"]
    for c in candidates:
        if c in df.columns:
            detected_target = c
            break
    if not detected_target and len(df.columns) > 1:
        detected_target = df.columns[-1]

    for col in df.columns:
        col_info: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "sampleValues": df[col].dropna().head(5).tolist(),
            "missingCount": int(df[col].isna().sum()),
            "uniqueCount": int(df[col].nunique()),
        }
        if is_numeric_dtype(df[col]):
            col_info["min"] = (
                float(df[col].min()) if pd.notna(df[col].min()) else None
            )
            col_info["max"] = (
                float(df[col].max()) if pd.notna(df[col].max()) else None
            )
            col_info["mean"] = (
                float(df[col].mean()) if pd.notna(df[col].mean()) else None
            )
        columns.append(col_info)

    return {
        "rowCount": len(df),
        "columnCount": len(df.columns),
        "columns": columns,
        "targetColumn": detected_target,
        "columnNames": list(df.columns),
    }


def run_evaluate(
    dataset_file: BinaryIO,
    dataset_name: str,
    sensitive_column: str,
    target_column: Optional[str] = None,
    evaluation_type: str = "model",
    model_file: Optional[BinaryIO] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Execute evaluation for any general AI model or dataset benchmark."""
    with temp_upload(dataset_file, dataset_name, MAX_DATASET_BYTES) as csv_path:
        df = pd.read_csv(csv_path)
        sensitive_column = validate_sensitive_column(df, sensitive_column)

        if evaluation_type == "dataset":
            counts = df[sensitive_column].value_counts()
            if counts.empty:
                raise ValueError("Sensitive column contains no data")
            balance = float(counts.min() / counts.max())
            return {
                "ethicsScore": round(float(balance * 10), 1),
                "fairnessScore": 0,
                "integrityScore": round(float(balance * 10), 1),
                "transparencyScore": 9.0,
                "biasDetected": balance < 0.5,
                "riskLevel": (
                    "High" if balance < 0.3 else "Medium" if balance < 0.7 else "Low"
                ),
                "framework": None,
                "isClassification": False,
                "metrics": {
                    "demographicParityDifference": 0,
                    "equalizedOddsDifference": 0,
                    "disparateImpact": round(balance, 3),
                    "robustness": 0,
                    "privacy": 0,
                },
                "groupPerformance": {
                    str(k): {
                        "count": int(v),
                        "percentage": float(v / len(df)),
                    }
                    for k, v in counts.items()
                },
                "recommendations": [
                    "Collect more balanced dataset samples for underrepresented demographic groups."
                ],
                "compliance": {
                    "frameworks": [],
                    "modelCard": {
                        "intendedUse": "",
                        "limitations": "",
                        "fairnessPhilosophy": "",
                    },
                },
                "chartData": [
                    {
                        "group": str(k),
                        "value": round(float(v / len(df) * 100), 1),
                        "label": "Representation",
                    }
                    for k, v in counts.items()
                ],
            }

        if not model_file or not model_name:
            raise ValueError("Model file artifact is required for model evaluation")

        validate_model_filename(model_name)
        with temp_upload(model_file, model_name, MAX_MODEL_BYTES) as model_path:
            model, framework = load_model(model_path, model_name)
            X, y_true, target_name = extract_target(df, target_column)
            if y_true is None:
                raise ValueError(
                    "Dataset must contain a valid target outcome column for model evaluation."
                )

            logger.info("Executing prediction on %d samples (framework: %s)...", len(X), framework)
            y_pred = predict_model(model, X, framework)
            fair = calculate_fairness_metrics(y_true, y_pred, df[sensitive_column])
            robustness_score = audit_robustness(model, X, framework)
            privacy_score = audit_privacy(y_pred, df[sensitive_column])

            f_score = (
                10 * (1 - min(1, fair["spd"])) * 0.5
                + 10 * min(1, fair["di"]) * 0.5
            )
            r_score = robustness_score * 10
            p_score = privacy_score * 10
            ethics_score = (f_score * 0.4) + (r_score * 0.3) + (p_score * 0.3)
            mitigation_rec = generate_mitigation_plan(fair)
            compliance = generate_compliance_report(ethics_score, fair)

            logger.info("Audit complete — Ethics Score: %.1f (framework: %s)", ethics_score, framework)

            return {
                "ethicsScore": round(float(ethics_score), 1),
                "fairnessScore": round(float(f_score), 1),
                "integrityScore": round(float(r_score), 1),
                "transparencyScore": round(float(p_score), 1),
                "biasDetected": bool(fair["spd"] > 0.1),
                "riskLevel": (
                    "High" if ethics_score < 6 else "Medium" if ethics_score < 8 else "Low"
                ),
                "framework": framework,
                "targetColumn": target_name,
                "isClassification": True,
                "metrics": {
                    "demographicParityDifference": round(float(fair["spd"]), 3),
                    "equalizedOddsDifference": round(float(fair["eod"]), 3),
                    "disparateImpact": round(float(fair["di"]), 3),
                    "robustness": round(float(robustness_score), 3),
                    "privacy": round(float(privacy_score), 3),
                },
                "groupPerformance": {
                    str(k): {
                        "selectionRate": float(v["selection_rate"]),
                        "accuracy": float(v["accuracy"]),
                        "count": int(v["count"]),
                    }
                    for k, v in fair["group_metrics"].items()
                },
                "recommendations": [mitigation_rec]
                + compliance["modelCard"]["limitations"].split(". "),
                "compliance": compliance,
                "chartData": [
                    {
                        "group": str(k),
                        "value": round(float(v["selection_rate"] * 100), 1),
                        "label": "Selection Rate",
                    }
                    for k, v in fair["group_metrics"].items()
                ],
            }


def _load_model_and_data(
    model_file: BinaryIO,
    model_name: str,
    dataset_file: BinaryIO,
    dataset_name: str,
    sensitive_column: str,
    target_column: Optional[str] = None,
) -> tuple[Any, str, pd.DataFrame, pd.DataFrame, Any, str]:
    validate_model_filename(model_name)
    with temp_upload(dataset_file, dataset_name, MAX_DATASET_BYTES) as csv_path:
        df = pd.read_csv(csv_path)
        sensitive_column = validate_sensitive_column(df, sensitive_column)
        with temp_upload(model_file, model_name, MAX_MODEL_BYTES) as model_path:
            model, framework = load_model(model_path, model_name)
            X, y_true, _ = extract_target(df, target_column)
            return model, framework, df, X, y_true, sensitive_column


def run_deep_audit(
    model_file: BinaryIO,
    model_name: str,
    dataset_file: BinaryIO,
    dataset_name: str,
    sensitive_columns: list[str],
    audit_type: str = "intersectional",
    target_column: Optional[str] = None,
) -> dict[str, Any]:
    sensitive_columns = [c.strip() for c in sensitive_columns if c and c.strip()]
    if not sensitive_columns:
        raise ValueError("At least one sensitive column is required")

    if audit_type == "intersectional":
        if len(sensitive_columns) < 2:
            raise ValueError("Intersectional analysis requires at least 2 sensitive columns")
        model, framework, df, X, y_true, _ = _load_model_and_data(
            model_file, model_name, dataset_file, dataset_name, sensitive_columns[0], target_column
        )
        for col in sensitive_columns:
            if col not in df.columns:
                raise ValueError(f'Column "{col}" not found in dataset')
        if y_true is None:
            raise ValueError("Dataset must contain a valid target column")
        y_pred = predict_model(model, X, framework)
        return intersectional_analysis(y_true, y_pred, df[sensitive_columns], sensitive_columns)

    if audit_type == "calibration":
        model, framework, df, X, y_true, sens = _load_model_and_data(
            model_file, model_name, dataset_file, dataset_name, sensitive_columns[0], target_column
        )
        if y_true is None:
            raise ValueError("Dataset must contain a valid target column")
        y_pred_proba = predict_proba_model(model, X, framework)
        if getattr(y_pred_proba, "ndim", 1) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        return calibration_parity(y_true, y_pred_proba, df[sens])

    raise ValueError(f"Unknown audit type: {audit_type}")


def run_optimize(
    model_file: BinaryIO,
    model_name: str,
    dataset_file: BinaryIO,
    dataset_name: str,
    sensitive_column: str,
    optimize_type: str = "threshold",
    target_column: Optional[str] = None,
) -> dict[str, Any]:
    model, framework, df, X, y_true, sens = _load_model_and_data(
        model_file, model_name, dataset_file, dataset_name, sensitive_column, target_column
    )
    if y_true is None:
        raise ValueError("Dataset must contain a target outcome column for optimization")

    if optimize_type == "threshold":
        y_pred_proba = predict_proba_model(model, X, framework)
        if getattr(y_pred_proba, "ndim", 1) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        return tune_thresholds(y_true, y_pred_proba, df[sens])

    if optimize_type == "retraining":
        if sens not in X.columns:
            X_with_sens = X.copy()
            X_with_sens[sens] = df[sens].values
            return generate_retraining_suggestions(X_with_sens, sens)
        return generate_retraining_suggestions(X, sens)

    if optimize_type == "regression-fairness":
        y_pred = predict_model(model, X, framework)
        return calculate_regression_fairness(y_true, y_pred, df[sens])

    raise ValueError(f"Unknown optimization type: {optimize_type}")


def run_security_audit(
    model_file: BinaryIO,
    model_name: str,
    dataset_file: BinaryIO,
    dataset_name: str,
    sensitive_column: str,
    audit_type: str = "adversarial",
    target_column: Optional[str] = None,
) -> dict[str, Any]:
    model, framework, df, X, y_true, sens = _load_model_and_data(
        model_file, model_name, dataset_file, dataset_name, sensitive_column, target_column
    )
    if audit_type == "adversarial":
        return adversarial_robustness(model, X, epsilon=0.1, framework=framework)
    if audit_type == "differential-privacy":
        return differential_privacy_audit(model, X, y_true, df[sens], framework)
    if audit_type == "membership-inference":
        return membership_inference_attack(model, X, y_true, df[sens], framework)
    raise ValueError(f"Unknown security audit type: {audit_type}")
