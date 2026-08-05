"""
AI Shayak — UI HTML Badges and Render Components.
"""

from __future__ import annotations


def risk_badge(level: str) -> str:
    """Generate HTML risk level badge (Low, Medium, High)."""
    cls = {"Low": "good", "Medium": "warn", "High": "bad"}.get(level, "outline")
    return f'<span class="as-badge {cls}">{level} Risk</span>'


def status_badge(status: str) -> str:
    """Generate HTML compliance/status badge."""
    good = status in ("Compliant", "Ethical", "Fair", "Low")
    bad = status in ("Non-Compliant", "High", "Bias")
    cls = "good" if good else "bad" if bad else "warn"
    return f'<span class="as-badge {cls}">{status}</span>'
