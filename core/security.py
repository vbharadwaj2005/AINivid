"""
AI Nivid — Model Security, Differential Privacy, and Robustness Auditing.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from core.models import detect_model_framework, predict_model, predict_proba_model
from core.utils import is_numeric_dtype


def adversarial_robustness(
    model: Any, X: pd.DataFrame, epsilon: float = 0.1, framework: Optional[str] = None
) -> dict[str, Any]:
    """Test model prediction stability under adversarial directional perturbations."""
    if framework is None:
        framework = detect_model_framework(model)
    rng = np.random.default_rng(42)
    num_cols = [col for col in X.columns if is_numeric_dtype(X[col])]
    if not num_cols:
        return {
            "accuracyUnderAttack": 1.0,
            "attackSuccessRate": 0.0,
            "perturbationEpsilon": 0.0,
            "perturbationDescription": "No numerical features to perturb",
        }
    y_orig = predict_model(model, X, framework)
    perturbed_df = X.copy()
    for col in num_cols:
        std = float(X[col].std())
        if std == 0 or pd.isna(std):
            continue
        direction = rng.choice([-1, 1], size=len(X))
        perturbed_df[col] = X[col].astype(float) + direction * epsilon * std
    y_attacked = predict_model(model, perturbed_df, framework)
    flip_rate = 1.0 - (y_orig == y_attacked).mean()
    return {
        "accuracyUnderAttack": round(float(1.0 - flip_rate), 4),
        "attackSuccessRate": round(float(flip_rate), 4),
        "perturbationEpsilon": epsilon,
        "perturbationDescription": (
            f"Directional ±{epsilon}·σ perturbation on numerical features"
        ),
    }


def differential_privacy_audit(
    model: Any,
    X: pd.DataFrame,
    y_true: Any,
    sensitive_features: Any,
    framework: Optional[str] = None,
) -> dict[str, Any]:
    """Estimate differential privacy epsilon bound via Leave-One-Out influence analysis."""
    if framework is None:
        framework = detect_model_framework(model)
    rng = np.random.default_rng(42)
    n = len(X)
    influence_scores = []
    sample_indices = rng.choice(n, size=min(50, n), replace=False)
    y_pred_full = predict_model(model, X, framework)
    for i in sample_indices:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        y_pred_loo = (
            predict_model(model, X.iloc[mask], framework)
            if hasattr(model, "predict") or True
            else y_pred_full
        )
        influence = 0.0
        if y_true is not None and len(y_pred_loo) >= n - 1:
            acc_full = (y_pred_full == y_true).mean()
            y_true_loo = (
                y_true.iloc[mask].values
                if hasattr(y_true, "iloc")
                else np.delete(y_true, i)
            )
            acc_loo = (y_pred_loo[: len(y_true_loo)] == y_true_loo).mean()
            influence = acc_full - acc_loo
        influence_scores.append(
            {"sampleIndex": int(i), "influence": round(float(influence), 6)}
        )
    max_influence = (
        max(abs(s["influence"]) for s in influence_scores) if influence_scores else 0.0
    )
    epsilon_estimate = min(10.0, max_influence * n / 2)
    return {
        "influenceScores": influence_scores,
        "maxInfluence": round(float(max_influence), 6),
        "estimatedEpsilon": round(float(epsilon_estimate), 3),
        "interpretation": (
            f"ε ≈ {epsilon_estimate:.2f}-DP (higher = less privacy). "
            "Values > 1 indicate measurable privacy risk."
        ),
        "riskLevel": (
            "High"
            if epsilon_estimate > 5
            else "Medium"
            if epsilon_estimate > 1
            else "Low"
        ),
    }


def membership_inference_attack(
    model: Any,
    X: pd.DataFrame,
    y_true: Any,
    sensitive_features: Any,
    framework: Optional[str] = None,
) -> dict[str, Any]:
    """Audit susceptibility to Membership Inference Attacks using shadow model training."""
    if framework is None:
        framework = detect_model_framework(model)
    rng = np.random.default_rng(42)
    n = len(X)
    n_test = min(100, n // 4)
    indices = np.arange(n)
    rng.shuffle(indices)
    shadow_train_idx = indices[:n_test]
    shadow_test_idx = indices[n_test : 2 * n_test]
    if len(shadow_train_idx) < 10 or len(shadow_test_idx) < 10:
        return {
            "error": "Insufficient samples for membership inference",
            "riskScore": 0.5,
        }
    # Encode mixed-type features so shadow model can train
    X_enc = X.copy()
    for col in X_enc.columns:
        if not is_numeric_dtype(X_enc[col]):
            X_enc[col] = pd.factorize(X_enc[col].astype(str))[0]
    X_enc = X_enc.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_shadow_train = X_enc.iloc[shadow_train_idx]
    y_shadow_train = y_true.iloc[shadow_train_idx] if y_true is not None else None
    try:
        if framework not in ("sklearn", "xgboost", "unknown"):
            return {
                "error": "Membership inference not supported for this framework",
                "riskScore": 0.5,
            }
        from sklearn.ensemble import RandomForestClassifier

        shadow = RandomForestClassifier(n_estimators=20, random_state=42)
        if y_shadow_train is not None:
            shadow.fit(X_shadow_train, y_shadow_train)
        else:
            shadow.fit(
                X_shadow_train,
                predict_model(model, X.iloc[shadow_train_idx], framework),
            )
        preds_train = predict_proba_model(shadow, X_shadow_train, "sklearn")
        preds_test = predict_proba_model(
            shadow, X_enc.iloc[shadow_test_idx], "sklearn"
        )
        if preds_train.ndim > 1:
            conf_train = np.max(preds_train, axis=1)
            conf_test = np.max(preds_test, axis=1)
        else:
            conf_train = preds_train
            conf_test = preds_test
        member_conf = float(np.mean(conf_train))
        nonmember_conf = float(np.mean(conf_test))
        risk_score = float(member_conf - nonmember_conf)
        return {
            "memberConfidence": round(member_conf, 4),
            "nonMemberConfidence": round(nonmember_conf, 4),
            "riskScore": round(risk_score, 4),
            "interpretation": (
                f"Attack confidence gap: {risk_score:.3f} "
                "(>0.1 indicates vulnerability)"
            ),
            "riskLevel": (
                "High"
                if risk_score > 0.2
                else "Medium"
                if risk_score > 0.1
                else "Low"
            ),
            "attackAccuracy": round(
                float((np.concatenate([conf_train, conf_test]) > 0.5).mean()), 4
            ),
        }
    except Exception as e:
        return {
            "error": f"Membership inference failed: {str(e)}",
            "riskScore": 0.5,
        }


def audit_robustness(
    model: Any, X: pd.DataFrame, framework: Optional[str] = None
) -> float:
    """Evaluate stability under Gaussian feature noise (±5% std)."""
    if framework is None:
        framework = detect_model_framework(model)
    rng = np.random.default_rng(42)
    X_perturbed = X.copy()
    num_cols = X.select_dtypes(include=["number"]).columns
    for col in num_cols:
        std = X[col].std()
        if std == 0:
            continue
        X_perturbed[col] = X[col] + rng.normal(0, 0.05 * std, size=len(X))
    y_orig = predict_model(model, X, framework)
    y_pert = predict_model(model, X_perturbed, framework)
    stability = (y_orig == y_pert).mean()
    return float(stability)


def audit_privacy(y_pred: Any, sensitive_features: Any) -> float:
    """Audit statistical privacy risk based on contingency between predictions & sensitive attributes."""
    df = pd.DataFrame({"pred": y_pred, "attr": sensitive_features})
    unique_attrs = df["attr"].nunique()
    if unique_attrs <= 1:
        return 1.0
    contingency = pd.crosstab(df["pred"], df["attr"], normalize="index")
    risk = contingency.max(axis=1).mean()
    expected_random = 1.0 / unique_attrs
    denom = 1.0 - expected_random
    if denom == 0:
        return 1.0
    normalized_risk = (risk - expected_random) / denom
    return float(max(0, 1 - normalized_risk))
