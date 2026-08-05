"""
AI Shayak — Regulatory Compliance, Mitigation Planning, and Report Generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def generate_mitigation_plan(fairness_results: dict[str, Any]) -> str:
    privileged = fairness_results["privileged"]
    unprivileged = fairness_results["unprivileged"]
    priv_tpr = fairness_results["group_metrics"][privileged]["tpr"]
    unpriv_tpr = fairness_results["group_metrics"][unprivileged]["tpr"]
    diff = priv_tpr - unpriv_tpr
    if diff > 0.05:
        return (
            f"Lower the decision threshold for group '{unprivileged}' by "
            f"{diff / 2:.2f} to equalize True Positive Rates across demographic groups."
        )
    return "No urgent decision threshold mitigation required."


def generate_compliance_report(
    ethics_score: float, fairness_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Generate regulatory AI governance compliance audit for general AI model evaluation."""
    risk_level = (
        "High"
        if fairness_metrics["spd"] > 0.2 or ethics_score < 5
        else "Limited"
        if ethics_score < 8
        else "Minimal"
    )
    return {
        "frameworks": [
            {
                "name": "EU AI Act",
                "status": "Non-Compliant" if risk_level == "High" else "Compliant",
                "requirement": "Article 10 (Data & Governance) - Bias Mitigation",
                "details": (
                    "High-risk AI systems must implement bias detection, governance, and audit logs."
                ),
            },
            {
                "name": "NITI Aayog (India)",
                "status": "Needs Review" if risk_level != "Minimal" else "Ethical",
                "requirement": "Principle of Equality & Non-Discrimination",
                "details": (
                    "AI systems must avoid unfair demographic disparities across protected subgroups."
                ),
            },
        ],
        "modelCard": {
            "intendedUse": (
                "General AI model inference, automated decisioning, and bias benchmarking."
            ),
            "limitations": (
                "Evaluated against uploaded benchmark dataset. Generalization and fairness bounds "
                "depend on demographic coverage and feature distribution."
            ),
            "fairnessPhilosophy": (
                "Demographic Parity and Equal Opportunity benchmarking across sensitive attributes."
            ),
        },
    }


def build_report_text(
    results: dict[str, Any],
    advanced: dict[str, Any],
    model_name: str,
    dataset_name: str,
    sensitive_column: str,
) -> str:
    """Format ethical AI audit report as plain text."""
    lines: list[str] = []
    add = lines.append
    r = results
    add("AI SHAYAK - ETHICAL AI & MODEL GOVERNANCE AUDIT REPORT")
    add("=====================================================")
    add("")
    add("Generated: " + datetime.now(timezone.utc).isoformat())
    add("Model Artifact: " + (model_name or "N/A"))
    add("Benchmark Dataset: " + (dataset_name or "N/A"))
    add("Sensitive Attribute: " + sensitive_column)
    add("Framework: " + str(r.get("framework") or "General"))
    add("")
    add("OVERALL STATUS")
    add("--------------")
    add(f"Ethics Score: {r.get('ethicsScore')}/10")
    add(f"Risk Level: {r.get('riskLevel')}")
    add(f"Fairness Score: {r.get('fairnessScore')}/10")
    add(f"Integrity Score: {r.get('integrityScore')}/10")
    add(f"Transparency Score: {r.get('transparencyScore')}/10")
    add("")
    metrics = r.get("metrics") or {}
    add("FAIRNESS METRICS")
    add("----------------")
    add(f"Demographic Parity Difference: {metrics.get('demographicParityDifference')} (ideal: 0)")
    add(f"Equalized Odds Difference: {metrics.get('equalizedOddsDifference')} (ideal: 0)")
    add(f"Disparate Impact Ratio: {metrics.get('disparateImpact')} (ideal: 1.0, threshold: 0.8)")
    rob = metrics.get("robustness") or 0
    priv = metrics.get("privacy") or 0
    add(f"Robustness: {rob * 100:.1f}%")
    add(f"Privacy: {priv * 100:.1f}%")
    add("")
    gp = r.get("groupPerformance") or {}
    if gp:
        add("GROUP PERFORMANCE")
        add("-----------------")
        for g, d in gp.items():
            sr = d.get("selectionRate")
            acc = d.get("accuracy")
            sr_s = f"{sr:.3f}" if isinstance(sr, (int, float)) else "N/A"
            acc_s = f"{(acc or 0) * 100:.1f}%" if acc is not None else "N/A"
            add(f"  {g}: Selection Rate={sr_s}, Acc={acc_s}, Count={d.get('count')}")
        add("")
    add("RECOMMENDATIONS")
    add("---------------")
    for i, rec in enumerate(r.get("recommendations") or [], 1):
        add(f"{i}. {rec}")
    add("")
    add("REGULATORY COMPLIANCE")
    add("---------------------")
    for fw in (r.get("compliance") or {}).get("frameworks") or []:
        add(f"{fw.get('name')}: {fw.get('status')}")
        add(f"  {fw.get('requirement')} - {fw.get('details')}")
    add("")
    for key, val in (advanced or {}).items():
        if not val or val.get("error"):
            continue
        add(key.upper() + " ANALYSIS")
        add("=" * (len(key) + 10))
        if val.get("recommendation"):
            add("Recommended: " + str(val["recommendation"]))
        if val.get("interpretation"):
            add("Interpretation: " + str(val["interpretation"]))
        if val.get("riskLevel"):
            add("Risk Level: " + str(val["riskLevel"]))
        if val.get("recommended"):
            rec = val["recommended"]
            add(
                f"Optimal Threshold: {rec.get('threshold')} "
                f"(SPD={rec.get('spd')}, Acc={rec.get('accuracy')})"
            )
        if val.get("potentialBiasedFeatures"):
            add("Biased Features: " + ", ".join(val["potentialBiasedFeatures"]))
        if val.get("accuracyUnderAttack") is not None:
            add(f"Accuracy Under Attack: {val['accuracyUnderAttack'] * 100:.1f}%")
        if val.get("attackSuccessRate") is not None:
            add(f"Attack Success Rate: {val['attackSuccessRate'] * 100:.1f}%")
        if val.get("riskScore") is not None:
            add(f"Membership Inference Risk Score: {val['riskScore']}")
        if val.get("estimatedEpsilon") is not None:
            add(f"Estimated DP Epsilon: {val['estimatedEpsilon']}")
        add("")
    return "\n".join(lines)
