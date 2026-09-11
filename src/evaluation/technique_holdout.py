from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import load_config, project_path
from src.data.split import parse_labels, select_grouped_holdout
from src.evaluation.metrics import multilabel_metrics
from src.models.train_classical import build_pipeline, load_partitions


def run_stress_test(config_path: str = "configs/experiment.yaml") -> dict:
    """Evaluate the selected SVM on ATT&CK techniques absent from training."""
    config = load_config(config_path)
    partitions, main_binarizer, _ = load_partitions(config)
    pool = partitions["train"].drop(columns="split").reset_index(drop=True)
    binarizer = MultiLabelBinarizer(classes=main_binarizer.classes_)
    targets = binarizer.fit_transform(parse_labels(pool["tactics"]))
    stress_config = config["technique_holdout"]
    candidate = select_grouped_holdout(
        pool,
        targets,
        stress_config["test_size"],
        config["seed"] + stress_config["seed_offset"],
        attempts=stress_config["candidate_attempts"],
        group_column="technique_id",
    )

    train = pool.iloc[candidate.train_indices]
    holdout = pool.iloc[candidate.holdout_indices]
    experiment = {
        "name": "svm_word_char_c1",
        "family": "linear_svm",
        "features": "word_char",
        "C": 1.0,
        "class_weight": "balanced",
    }
    pipeline = build_pipeline(experiment, config)
    pipeline.fit(train["text"], targets[candidate.train_indices])
    predictions = pipeline.predict(holdout["text"])
    metrics = multilabel_metrics(
        targets[candidate.holdout_indices], predictions, binarizer.classes_.tolist()
    )

    result = {
        "purpose": "Stress test generalization to ATT&CK techniques absent from training",
        "selection_dataset": "main grouped-source training partition only",
        "model": "TF-IDF word+character features with Linear SVM",
        "selection_seed": candidate.seed,
        "selection_balance_score": candidate.score,
        "train_examples": len(train),
        "holdout_examples": len(holdout),
        "train_techniques": int(train["technique_id"].nunique()),
        "holdout_techniques": int(holdout["technique_id"].nunique()),
        "technique_overlap": len(set(train["technique_id"]) & set(holdout["technique_id"])),
        "source_overlap": len(set(train["source_id"]) & set(holdout["source_id"])),
        "metrics": metrics,
    }
    output = project_path("artifacts/metrics/technique_holdout_metrics.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unseen-technique stress test")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    np.random.seed(load_config(args.config)["seed"])
    print(json.dumps(run_stress_test(args.config), indent=2))


if __name__ == "__main__":
    main()
