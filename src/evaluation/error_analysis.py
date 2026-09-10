from __future__ import annotations

import argparse
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import load_config, project_path
from src.models.train_classical import load_partitions


def analyze_errors(config_path: str = "configs/experiment.yaml") -> dict:
    config = load_config(config_path)
    partitions, binarizer, targets = load_partitions(config)
    test = partitions["test"]
    y_true = targets["test"]
    labels = binarizer.classes_.tolist()
    artifact = joblib.load(project_path("artifacts/models/linear_svm.joblib"))
    scores = artifact["pipeline"].decision_function(test["text"])
    y_pred = (scores >= 0).astype(int)

    wrong = np.any(y_true != y_pred, axis=1)
    rows = []
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    for index in np.flatnonzero(wrong):
        missed = [labels[column] for column in np.flatnonzero((y_true[index] == 1) & (y_pred[index] == 0))]
        extra = [labels[column] for column in np.flatnonzero((y_true[index] == 0) & (y_pred[index] == 1))]
        for true_label in missed:
            for predicted_label in extra or ["<none>"]:
                confusion_pairs[(true_label, predicted_label)] += 1
        rows.append(
            {
                "text": test.iloc[index]["text"],
                "true_labels": test.iloc[index]["tactics"],
                "predicted_labels": "|".join(labels[column] for column in np.flatnonzero(y_pred[index])),
                "missed_labels": "|".join(missed),
                "extra_labels": "|".join(extra),
                "largest_decision_margin": float(np.max(scores[index])),
                "source_name": test.iloc[index]["source_name"],
                "source_type": test.iloc[index]["source_type"],
                "technique_id": test.iloc[index]["technique_id"],
                "technique_name": test.iloc[index]["technique_name"],
            }
        )

    error_frame = pd.DataFrame(rows).sort_values("largest_decision_margin", ascending=False)
    error_frame.to_csv(project_path("artifacts/metrics/misclassified_svm_examples.csv"), index=False)
    multilabel_mask = y_true.sum(axis=1) > 1
    summary = {
        "test_examples": len(test),
        "examples_with_any_label_error": int(wrong.sum()),
        "examples_with_no_predicted_label": int((y_pred.sum(axis=1) == 0).sum()),
        "samples_f1_single_label_records": float(f1_score(y_true[~multilabel_mask], y_pred[~multilabel_mask], average="samples", zero_division=0)),
        "samples_f1_multilabel_records": float(f1_score(y_true[multilabel_mask], y_pred[multilabel_mask], average="samples", zero_division=0)),
        "top_missed_to_extra_pairs": [
            {"missed": missed, "extra": extra, "count": count}
            for (missed, extra), count in confusion_pairs.most_common(12)
        ],
        "errors_by_source_type": error_frame["source_type"].value_counts().to_dict(),
    }
    project_path("artifacts/metrics/error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze grouped test-set errors for the selected SVM")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    print(json.dumps(analyze_errors(args.config), indent=2))


if __name__ == "__main__":
    main()
