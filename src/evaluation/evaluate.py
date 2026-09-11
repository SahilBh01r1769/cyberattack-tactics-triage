from __future__ import annotations

import argparse
import json
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import load_config, project_path
from src.evaluation.metrics import multilabel_metrics
from src.models.train_classical import load_partitions


def _read_metrics(filename: str) -> dict:
    return json.loads(project_path(f"artifacts/metrics/{filename}").read_text(encoding="utf-8"))


def transformer_comparison_row(metrics: dict) -> dict | None:
    """Return a comparison row only for a completed full-data run."""
    if metrics.get("smoke_limit") is not None:
        return None
    test_metrics = metrics.get("validation_tuned_test_metrics")
    if not test_metrics:
        return None
    return {"model": f"DistilBERT ({metrics['model_name']})", **test_metrics}


def build_model_comparison(config: dict, partitions: dict, targets: dict, binarizer: MultiLabelBinarizer) -> pd.DataFrame:
    most_common = Counter(partitions["train"]["tactics"]).most_common(1)[0][0]
    baseline_row = binarizer.transform([most_common.split("|")])[0]
    baseline = multilabel_metrics(
        targets["test"], np.tile(baseline_row, (len(partitions["test"]), 1)), binarizer.classes_.tolist()
    )
    lr = _read_metrics("logistic_regression_test_metrics.json")
    svm = _read_metrics("linear_svm_test_metrics.json")
    calibrated = _read_metrics("confidence_metrics.json")["thresholded_test_metrics"]
    rows = [
        {"model": "Most frequent label set", **baseline},
        {"model": "TF-IDF + Logistic Regression", **lr},
        {"model": "Calibrated TF-IDF + LR", **calibrated},
        {"model": "TF-IDF + Linear SVM", **svm},
    ]
    transformer_path = project_path("artifacts/metrics/transformer_metrics.json")
    if transformer_path.exists():
        transformer_row = transformer_comparison_row(json.loads(transformer_path.read_text(encoding="utf-8")))
        if transformer_row:
            rows.append(transformer_row)
    columns = ["model", "macro_f1", "micro_f1", "weighted_f1", "samples_f1", "hamming_loss", "subset_accuracy"]
    return pd.DataFrame(rows)[columns]


def label_confusion_counts(targets: np.ndarray, predictions: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    for column, label in enumerate(labels):
        truth = targets[:, column].astype(bool)
        predicted = predictions[:, column].astype(bool)
        rows.append(
            {
                "tactic": label,
                "true_negative": int((~truth & ~predicted).sum()),
                "false_positive": int((~truth & predicted).sum()),
                "false_negative": int((truth & ~predicted).sum()),
                "true_positive": int((truth & predicted).sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_dataset_overview(frame: pd.DataFrame, splits: pd.DataFrame) -> None:
    figure_dir = project_path("artifacts/figures")
    labels = Counter(label for value in frame["tactics"] for label in value.split("|"))
    lengths = frame["text"].str.split().str.len()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    pd.Series(labels).sort_values().plot.barh(ax=axes[0], color="#3b6ea8")
    axes[0].set(title="Tactic frequency", xlabel="Procedure examples", ylabel="")
    axes[1].hist(lengths, bins=35, color="#4f8f70", edgecolor="white")
    axes[1].set(title="Text length", xlabel="Words", ylabel="Examples")
    splits["split"].value_counts().reindex(["train", "validation", "test"]).plot.bar(ax=axes[2], color="#a06b3b")
    axes[2].set(title="Grouped split sizes", xlabel="", ylabel="Examples")
    axes[2].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figure_dir / "dataset_overview.png", dpi=160)
    plt.close(fig)


def _plot_model_comparison(comparison: pd.DataFrame) -> None:
    figure_dir = project_path("artifacts/figures")
    metrics = comparison.set_index("model")[["macro_f1", "micro_f1", "samples_f1"]]
    axis = metrics.plot.bar(figsize=(9, 5), color=["#3b6ea8", "#4f8f70", "#a06b3b"])
    axis.set(title="Grouped test-set performance", ylabel="F1", xlabel="", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(figure_dir / "model_comparison.png", dpi=160)
    plt.close(axis.figure)


def _plot_label_errors(confusion: pd.DataFrame) -> None:
    plot_frame = confusion.set_index("tactic")[["false_positive", "false_negative"]]
    axis = plot_frame.plot.bar(figsize=(11, 5), color=["#b05a5a", "#6b75a8"])
    axis.set(title="Linear SVM errors by tactic", ylabel="Test examples", xlabel="")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(project_path("artifacts/figures/label_error_counts.png"), dpi=160)
    plt.close(axis.figure)


def evaluate(config_path: str = "configs/experiment.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    partitions, binarizer, targets = load_partitions(config)
    comparison = build_model_comparison(config, partitions, targets, binarizer)
    metrics_dir = project_path("artifacts/metrics")
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)

    svm_artifact = joblib.load(project_path("artifacts/models/linear_svm.joblib"))
    predictions = svm_artifact["pipeline"].predict(partitions["test"]["text"])
    confusion = label_confusion_counts(targets["test"], predictions, binarizer.classes_.tolist())
    confusion.to_csv(metrics_dir / "label_confusion_counts.csv", index=False)

    raw_frame = pd.read_csv(project_path(config["dataset"]["processed_path"]))
    split_frame = pd.read_csv(project_path(config["dataset"]["split_path"]))
    _plot_dataset_overview(raw_frame, split_frame)
    _plot_model_comparison(comparison)
    _plot_label_errors(confusion)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final classical-model evaluation artifacts")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    print(evaluate(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
