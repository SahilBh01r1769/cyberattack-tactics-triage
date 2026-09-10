from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score

from src.config import load_config, project_path
from src.evaluation.metrics import multilabel_metrics
from src.models.train_classical import load_partitions


@dataclass
class PerLabelPlattCalibrator:
    """Independent sigmoid calibrators fitted to validation decision scores."""

    models: list[LogisticRegression]

    @classmethod
    def fit(cls, scores: np.ndarray, targets: np.ndarray, seed: int) -> "PerLabelPlattCalibrator":
        models = []
        for column in range(targets.shape[1]):
            model = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
            model.fit(scores[:, [column]], targets[:, column])
            models.append(model)
        return cls(models)

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [model.predict_proba(scores[:, [column]])[:, 1] for column, model in enumerate(self.models)]
        )


def expected_calibration_error(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    errors = []
    for column in range(targets.shape[1]):
        label_errors = []
        for lower in np.linspace(0, 1, bins, endpoint=False):
            upper = lower + 1 / bins
            mask = (probabilities[:, column] >= lower) & (probabilities[:, column] < upper)
            if mask.any():
                label_errors.append(mask.mean() * abs(targets[mask, column].mean() - probabilities[mask, column].mean()))
        errors.append(sum(label_errors))
    return float(np.mean(errors))


def calibration_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    brier_scores = [brier_score_loss(targets[:, column], probabilities[:, column]) for column in range(targets.shape[1])]
    return {
        "macro_brier_score": float(np.mean(brier_scores)),
        "macro_expected_calibration_error": expected_calibration_error(targets, probabilities),
    }


def choose_label_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    candidates = np.linspace(0.10, 0.90, 33)
    thresholds = []
    for column in range(targets.shape[1]):
        scores = [f1_score(targets[:, column], probabilities[:, column] >= value, zero_division=0) for value in candidates]
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return np.asarray(thresholds)


def routing_confidence(probabilities: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Confidence is the strongest predicted tactic probability, or the strongest candidate if none pass."""
    masked = np.where(predictions == 1, probabilities, -1.0)
    positive_max = masked.max(axis=1)
    overall_max = probabilities.max(axis=1)
    return np.where(positive_max >= 0, positive_max, overall_max)


def route_predictions(confidences: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(confidences >= threshold, "auto_route", "analyst_review")


def confidence_tradeoff(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    confidence = routing_confidence(probabilities, predictions)
    rows = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        row = {"threshold": threshold, "coverage": float(accepted.mean()), "accepted_samples": int(accepted.sum())}
        if accepted.any():
            metrics = multilabel_metrics(targets[accepted], predictions[accepted], labels)
            row.update({"macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"], "samples_f1": metrics["samples_f1"], "subset_accuracy": metrics["subset_accuracy"]})
        else:
            row.update({"macro_f1": None, "micro_f1": None, "samples_f1": None, "subset_accuracy": None})
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_bins(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    flat_targets = targets.ravel()
    flat_probabilities = probabilities.ravel()
    rows = []
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (flat_probabilities >= lower) & (flat_probabilities < upper)
        if mask.any():
            rows.append({"lower": lower, "upper": upper, "mean_confidence": flat_probabilities[mask].mean(), "observed_frequency": flat_targets[mask].mean(), "count": int(mask.sum())})
    return pd.DataFrame(rows)


def _save_figures(tradeoff: pd.DataFrame, reliability: pd.DataFrame) -> None:
    figure_dir = project_path("artifacts/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(tradeoff["threshold"], tradeoff["coverage"], marker="o", label="Coverage")
    axis.plot(tradeoff["threshold"], tradeoff["samples_f1"], marker="o", label="Accepted samples F1")
    axis.set(xlabel="Routing confidence threshold", ylabel="Score", ylim=(0, 1), title="Confidence-aware routing tradeoff")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "confidence_coverage.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Ideal")
    axis.plot(reliability["mean_confidence"], reliability["observed_frequency"], marker="o", label="Calibrated LR")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1), title="Reliability plot")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "reliability_plot.png", dpi=160)
    plt.close(fig)


def calibrate(config_path: str = "configs/experiment.yaml") -> dict:
    config = load_config(config_path)
    artifact = joblib.load(project_path("artifacts/models/logistic_regression.joblib"))
    pipeline = artifact["pipeline"]
    partitions, binarizer, targets = load_partitions(config)
    labels = binarizer.classes_.tolist()

    validation_scores = pipeline.decision_function(partitions["validation"]["text"])
    test_scores = pipeline.decision_function(partitions["test"]["text"])
    raw_test_probabilities = pipeline.predict_proba(partitions["test"]["text"])
    calibrator = PerLabelPlattCalibrator.fit(validation_scores, targets["validation"], config["seed"])
    validation_probabilities = calibrator.predict_proba(validation_scores)
    test_probabilities = calibrator.predict_proba(test_scores)
    label_thresholds = choose_label_thresholds(targets["validation"], validation_probabilities)
    test_predictions = (test_probabilities >= label_thresholds).astype(int)

    model_metrics = multilabel_metrics(targets["test"], test_predictions, labels)
    metrics = {
        "method": "per-label Platt scaling on validation partition",
        "raw_probability_calibration": calibration_metrics(targets["test"], raw_test_probabilities),
        "platt_calibration": calibration_metrics(targets["test"], test_probabilities),
        "label_thresholds": dict(zip(labels, label_thresholds.tolist())),
        "thresholded_test_metrics": model_metrics,
    }
    metrics_dir = project_path("artifacts/metrics")
    (metrics_dir / "confidence_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tradeoff = confidence_tradeoff(targets["test"], test_predictions, test_probabilities, labels, config["confidence"]["thresholds"])
    tradeoff.to_csv(metrics_dir / "confidence_tradeoff.csv", index=False)
    reliability = reliability_bins(targets["test"], test_probabilities)
    reliability.to_csv(metrics_dir / "reliability_bins.csv", index=False)
    _save_figures(tradeoff, reliability)

    joblib.dump(
        {**artifact, "calibrator": calibrator, "label_thresholds": label_thresholds, "labels": labels},
        project_path("artifacts/models/confidence_logistic_regression.joblib"),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the logistic regression tactic classifier")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    metrics = calibrate(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

