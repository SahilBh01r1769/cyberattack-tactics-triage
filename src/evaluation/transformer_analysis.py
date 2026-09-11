from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import project_path


def parse_label_values(series: pd.Series) -> list[list[str]]:
    return [str(value).split("|") if pd.notna(value) and str(value) else [] for value in series]


def summarize_prediction_errors(predictions: pd.DataFrame, labels: list[str]) -> tuple[dict, pd.DataFrame]:
    binarizer = MultiLabelBinarizer(classes=labels)
    y_true = binarizer.fit_transform(parse_label_values(predictions["true_labels"]))
    y_pred = binarizer.transform(parse_label_values(predictions["predicted_labels"]))
    wrong = np.any(y_true != y_pred, axis=1)
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    missed_counts: Counter[str] = Counter()
    extra_counts: Counter[str] = Counter()

    for index in np.flatnonzero(wrong):
        missed = [labels[column] for column in np.flatnonzero((y_true[index] == 1) & (y_pred[index] == 0))]
        extra = [labels[column] for column in np.flatnonzero((y_true[index] == 0) & (y_pred[index] == 1))]
        missed_counts.update(missed)
        extra_counts.update(extra)
        for true_label in missed:
            for predicted_label in extra or ["<none>"]:
                confusion_pairs[(true_label, predicted_label)] += 1

    multilabel_mask = y_true.sum(axis=1) > 1
    summary = {
        "test_examples": len(predictions),
        "examples_with_any_label_error": int(wrong.sum()),
        "examples_with_no_predicted_label": int((y_pred.sum(axis=1) == 0).sum()),
        "samples_f1_single_label_records": float(
            f1_score(y_true[~multilabel_mask], y_pred[~multilabel_mask], average="samples", zero_division=0)
        ),
        "samples_f1_multilabel_records": float(
            f1_score(y_true[multilabel_mask], y_pred[multilabel_mask], average="samples", zero_division=0)
        ),
        "most_missed_labels": dict(missed_counts.most_common()),
        "most_extra_labels": dict(extra_counts.most_common()),
        "top_missed_to_extra_pairs": [
            {"missed": missed, "extra": extra, "count": count}
            for (missed, extra), count in confusion_pairs.most_common(12)
        ],
    }
    error_frame = predictions.loc[wrong].copy().sort_values("top_confidence", ascending=False)
    return summary, error_frame


def _plot_training_history(metrics: dict) -> None:
    evaluations = [row for row in metrics["log_history"] if "eval_macro_f1" in row]
    frame = pd.DataFrame(evaluations)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(frame["epoch"], frame["eval_macro_f1"], marker="o", label="Validation macro F1")
    axis.plot(frame["epoch"], frame["eval_micro_f1"], marker="o", label="Validation micro F1")
    axis.set(xlabel="Epoch", ylabel="F1", xticks=frame["epoch"], ylim=(0, 1), title="DistilBERT validation history")
    axis.grid(alpha=0.25)
    loss_axis = axis.twinx()
    loss_axis.plot(frame["epoch"], frame["eval_loss"], marker="s", linestyle="--", color="#a65f4b", label="Validation loss")
    loss_axis.set_ylabel("Loss")
    handles, names = axis.get_legend_handles_labels()
    loss_handles, loss_names = loss_axis.get_legend_handles_labels()
    axis.legend(handles + loss_handles, names + loss_names, loc="center right")
    fig.tight_layout()
    fig.savefig(project_path("artifacts/figures/transformer_training_history.png"), dpi=160)
    plt.close(fig)


def _per_label_comparison(metrics: dict) -> pd.DataFrame:
    svm = json.loads(project_path("artifacts/metrics/linear_svm_test_metrics.json").read_text(encoding="utf-8"))
    transformer_scores = metrics["validation_tuned_test_metrics"]["per_label_f1"]
    frame = pd.DataFrame(
        {
            "tactic": list(transformer_scores),
            "linear_svm_f1": [svm["per_label_f1"][label] for label in transformer_scores],
            "distilbert_f1": list(transformer_scores.values()),
        }
    )
    frame["distilbert_minus_svm"] = frame["distilbert_f1"] - frame["linear_svm_f1"]
    frame.to_csv(project_path("artifacts/metrics/transformer_per_label_comparison.csv"), index=False)

    axis = frame.set_index("tactic")[["linear_svm_f1", "distilbert_f1"]].plot.bar(
        figsize=(11, 5), color=["#4f8f70", "#3b6ea8"]
    )
    axis.set(title="Per-tactic F1 on the grouped test set", ylabel="F1", xlabel="", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(["Linear SVM", "DistilBERT"], loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(project_path("artifacts/figures/transformer_per_label_comparison.png"), dpi=160)
    plt.close(axis.figure)
    return frame


def analyze() -> dict:
    metrics_path = project_path("artifacts/metrics/transformer_metrics.json")
    predictions_path = project_path("artifacts/metrics/transformer_test_predictions.csv")
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Run the transformer or unpack transformer_evidence.zip before analysis")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("smoke_limit") is not None:
        raise ValueError("Smoke-run output is not valid final evidence")
    predictions = pd.read_csv(predictions_path)
    labels = list(metrics["label_thresholds_selected_on_validation"])
    summary, errors = summarize_prediction_errors(predictions, labels)
    project_path("artifacts/metrics/transformer_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    errors.to_csv(project_path("artifacts/metrics/misclassified_transformer_examples.csv"), index=False)
    _plot_training_history(metrics)
    _per_label_comparison(metrics)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed DistilBERT test predictions")
    parser.parse_args()
    print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()

