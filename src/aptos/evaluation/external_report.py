"""Scientific, aggregate-only external validation summaries.

Undefined rates are JSON null, not zero or non-standard NaN tokens.
"""
from __future__ import annotations

import numpy as np

from aptos.evaluation.paired_uncertainty import _grades
from aptos.modeling.thresholds import (
    clinically_costly_errors,
    grade_metrics,
    per_class_recall,
    referable_confusion,
)


def _nullable(metrics: dict) -> dict:
    """Convert undefined floating metrics to explicit, strict JSON nulls."""
    return {
        name: (None if isinstance(value, (float, np.floating))
               and not np.isfinite(value) else value)
        for name, value in metrics.items()
    }


def summarize_external_grades(diagnosis, predictions) -> dict:
    """Summarize a fixed cohort without fabricating missing classes or rates."""
    actual = _grades(diagnosis, "diagnosis")
    predicted = _grades(predictions, "predictions")
    if len(actual) != len(predicted):
        raise ValueError("external grades and predictions require equal lengths")

    metrics = _nullable(grade_metrics(actual, predicted))
    metrics.update({
        f"ref_{key}": value for key, value in
        _nullable(referable_confusion(actual, predicted)).items()
    })
    metrics.update(clinically_costly_errors(actual, predicted))
    metrics["per_class_recall"] = {
        str(grade): value for grade, value in
        _nullable(per_class_recall(actual, predicted)).items()
    }
    metrics["predicted_distribution"] = {
        str(grade): int((predicted == grade).sum()) for grade in range(5)
    }
    metrics["true_distribution"] = {
        str(grade): int((actual == grade).sum()) for grade in range(5)
    }
    healthy = actual == 0
    count = int(healthy.sum())
    metrics["healthy_n"] = count
    metrics["healthy_called_healthy"] = float((predicted[healthy] == 0).mean()) if count else None
    metrics["healthy_called_referable"] = float((predicted[healthy] >= 2).mean()) if count else None
    return metrics


def metric_string(value, *, percent: bool = False) -> str:
    """Use N/A for undefined values while retaining original report precision."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.1%}" if percent else f"{number:.4f}"
