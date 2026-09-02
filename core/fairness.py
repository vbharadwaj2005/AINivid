"""
AI Nivid — Ethics, Bias, and Fairness Metrics Calculation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_fairness_metrics(
    y_true: Any, y_pred: Any, sensitive_features: Any
) -> dict[str, Any]:
    """Calculate Demographic Parity, Disparate Impact, Equalized Odds, and Equal Opportunity."""
    df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "group": sensitive_features}
    )
    groups = df["group"].unique()
    group_metrics: dict[str, dict[str, Any]] = {}

    for group in groups:
        group_df = df[df["group"] == group]
        tp = ((group_df["y_true"] == 1) & (group_df["y_pred"] == 1)).sum()
        fp = ((group_df["y_true"] == 0) & (group_df["y_pred"] == 1)).sum()
        tn = ((group_df["y_true"] == 0) & (group_df["y_pred"] == 0)).sum()
        fn = ((group_df["y_true"] == 1) & (group_df["y_pred"] == 0)).sum()
        selection_rate = group_df["y_pred"].mean()
        accuracy = (tp + tn) / len(group_df) if len(group_df) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        group_metrics[group] = {
            "selection_rate": float(selection_rate),
            "accuracy": float(accuracy),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "count": int(len(group_df)),
        }

    sorted_groups = sorted(
        group_metrics.items(), key=lambda x: x[1]["selection_rate"], reverse=True
    )
    privileged_group = sorted_groups[0][0]
    unprivileged_group = sorted_groups[-1][0]
    priv = group_metrics[privileged_group]
    unpriv = group_metrics[unprivileged_group]
    spd = priv["selection_rate"] - unpriv["selection_rate"]
    di = (
        unpriv["selection_rate"] / priv["selection_rate"]
        if priv["selection_rate"] > 0
        else 1.0
    )
    eod = abs(priv["tpr"] - unpriv["tpr"])
    aod = 0.5 * (abs(priv["fpr"] - unpriv["fpr"]) + abs(priv["tpr"] - unpriv["tpr"]))
    return {
        "spd": spd,
        "di": di,
        "eod": eod,
        "aod": aod,
        "group_metrics": group_metrics,
        "privileged": privileged_group,
        "unprivileged": unprivileged_group,
    }


def calculate_regression_fairness(
    y_true: Any, y_pred: Any, sensitive_features: Any
) -> dict[str, Any]:
    """Calculate mean prediction difference and MAE disparity for regression targets."""
    df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "group": sensitive_features}
    )
    groups = df["group"].unique()
    group_metrics: dict[str, dict[str, Any]] = {}
    overall_mean = y_pred.mean() if hasattr(y_pred, "mean") else float(np.mean(y_pred))
    for group in groups:
        g = df[df["group"] == group]
        group_metrics[group] = {
            "count": int(len(g)),
            "mean_prediction": float(g["y_pred"].mean()),
            "mean_actual": float(g["y_true"].mean()),
            "mae": float(abs(g["y_pred"] - g["y_true"]).mean()),
        }
    groups_list = sorted(group_metrics.keys())
    mean_diff = 0.0
    max_mae_gap = 0.0
    if len(groups_list) >= 2:
        vals = [group_metrics[g]["mean_prediction"] for g in groups_list]
        mean_diff = float(max(vals) - min(vals))
        mae_vals = [group_metrics[g]["mae"] for g in groups_list]
        max_mae_gap = float(max(mae_vals) - min(mae_vals))
    return {
        "group_metrics": group_metrics,
        "meanPredictionDifference": round(mean_diff, 4),
        "maxMaeDisparity": round(max_mae_gap, 4),
        "overallMean": float(overall_mean),
    }


def intersectional_analysis(
    y_true: Any, y_pred: Any, sensitive_columns_df: pd.DataFrame, columns: list[str]
) -> dict[str, Any]:
    """Perform intersectional bias evaluation combining multiple sensitive attributes."""
    combined = sensitive_columns_df[columns[0]].astype(str)
    for col in columns[1:]:
        combined = combined + "_" + sensitive_columns_df[col].astype(str)
    fair = calculate_fairness_metrics(y_true, y_pred, combined)
    return {
        "intersection_groups": combined.value_counts().to_dict(),
        "fairness": {
            "spd": fair["spd"],
            "di": fair["di"],
            "eod": fair["eod"],
            "aod": fair["aod"],
            "privileged": fair["privileged"],
            "unprivileged": fair["unprivileged"],
        },
        "group_metrics": {
            str(k): {
                "selectionRate": float(v["selection_rate"]),
                "accuracy": float(v["accuracy"]),
                "count": int(v["count"]),
            }
            for k, v in fair["group_metrics"].items()
        },
        "chartData": [
            {
                "group": str(k),
                "value": round(float(v["selection_rate"] * 100), 1),
                "label": "Selection Rate",
            }
            for k, v in fair["group_metrics"].items()
        ],
    }


def calibration_parity(
    y_true: Any, y_pred_proba: Any, sensitive_features: Any, n_bins: int = 10
) -> dict[str, Any]:
    """Audit calibration parity across demographic subgroups."""
    df = pd.DataFrame(
        {"y_true": y_true, "proba": y_pred_proba, "group": sensitive_features}
    )
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(n_bins)]
    df["bin"] = pd.cut(
        df["proba"], bins=bins, labels=bin_labels, include_lowest=True
    )
    groups = df["group"].unique()
    result: dict[str, Any] = {}
    for group in groups:
        g = df[df["group"] == group]
        cal = (
            g.groupby("bin", observed=True)
            .apply(
                lambda x: {
                    "meanPred": float(x["proba"].mean()),
                    "meanActual": float(x["y_true"].mean()),
                    "count": int(len(x)),
                },
                include_groups=False,
            )
            .to_dict()
        )
        result[str(group)] = cal
    cal_error = 0.0
    for _group, cal_data in result.items():
        for _b, v in cal_data.items():
            if v["count"] > 0:
                cal_error += abs(v["meanPred"] - v["meanActual"]) * v["count"]
    cal_error /= max(len(df), 1)
    return {
        "calibrationData": result,
        "calibrationError": round(float(cal_error), 4),
        "bins": bin_labels,
    }


def tune_thresholds(
    y_true: Any, y_pred_proba: Any, sensitive_features: Any
) -> dict[str, Any]:
    """Find optimal decision thresholds to balance fairness and accuracy."""
    thresholds = np.linspace(0.05, 0.95, 19)
    results = []
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        fair = calculate_fairness_metrics(y_true, y_pred_t, sensitive_features)
        acc = (y_pred_t == y_true).mean()
        results.append(
            {
                "threshold": round(float(t), 2),
                "spd": round(float(fair["spd"]), 4),
                "di": round(float(fair["di"]), 4),
                "accuracy": round(float(acc), 4),
                "biasDetected": bool(fair["spd"] > 0.1),
            }
        )
    best_fairness = min(results, key=lambda r: abs(r["spd"]))
    best_accuracy = max(results, key=lambda r: r["accuracy"])
    best_tradeoff = min(
        results, key=lambda r: abs(r["spd"]) * 2 + (1 - r["accuracy"])
    )
    return {
        "thresholds": results,
        "optimalFairness": best_fairness,
        "optimalAccuracy": best_accuracy,
        "recommended": best_tradeoff,
        "recommendation": (
            f"Recommended threshold: {best_tradeoff['threshold']} "
            f"(SPD={best_tradeoff['spd']}, Accuracy={best_tradeoff['accuracy']})"
        ),
    }


def generate_retraining_suggestions(
    X: pd.DataFrame, sensitive_column: str
) -> dict[str, Any]:
    """Generate recommendations for retraining, resampling, or reweighting."""
    suggestions = []
    sens_codes = pd.Series(
        pd.factorize(X[sensitive_column])[0], index=X.index
    )
    for col in X.select_dtypes(include=["number"]).columns[:10]:
        if col == sensitive_column:
            continue
        corr = abs(X[col].corr(sens_codes))
        suggestions.append(
            {
                "feature": col,
                "correlation_with_sensitive": round(float(corr), 3)
                if pd.notna(corr)
                else 0.0,
            }
        )
    suggestions.sort(key=lambda x: x["correlation_with_sensitive"], reverse=True)
    biased_features = [
        s["feature"]
        for s in suggestions[:5]
        if s["correlation_with_sensitive"] > 0.1
    ]
    return {
        "featureCorrelations": suggestions[:10],
        "potentialBiasedFeatures": biased_features,
        "resamplingSuggestion": (
            "Resampling: Upsample underrepresented groups or downsample "
            "overrepresented groups in the training data."
            if biased_features
            else "No significant feature-group correlations detected."
        ),
        "reweightingSuggestion": (
            "Reweighting: Assign higher sample weights to unprivileged group "
            "samples during training."
            if biased_features
            else "No reweighting needed based on feature analysis."
        ),
        "adversarialDebiasingSuggestion": (
            "For neural models: Add an adversarial branch that predicts the "
            "sensitive attribute from model embeddings, and train to minimize this."
        ),
    }
