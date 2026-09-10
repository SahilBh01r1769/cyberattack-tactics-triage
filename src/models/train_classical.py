from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from src.config import load_config, project_path
from src.evaluation.metrics import multilabel_metrics
from src.features.text_features import make_tfidf


LOGGER = logging.getLogger(__name__)


EXPERIMENTS = [
    {"name": "lr_word_unigram_c1", "family": "logistic_regression", "features": "word_unigram", "C": 1.0, "class_weight": "balanced"},
    {"name": "lr_word_bigram_c05", "family": "logistic_regression", "features": "word_bigram", "C": 0.5, "class_weight": "balanced"},
    {"name": "lr_word_bigram_c1", "family": "logistic_regression", "features": "word_bigram", "C": 1.0, "class_weight": "balanced"},
    {"name": "lr_word_bigram_c2", "family": "logistic_regression", "features": "word_bigram", "C": 2.0, "class_weight": "balanced"},
    {"name": "lr_word_bigram_c2_unweighted", "family": "logistic_regression", "features": "word_bigram", "C": 2.0, "class_weight": None},
    {"name": "lr_word_char_c2", "family": "logistic_regression", "features": "word_char", "C": 2.0, "class_weight": "balanced"},
    {"name": "svm_word_bigram_c05", "family": "linear_svm", "features": "word_bigram", "C": 0.5, "class_weight": "balanced"},
    {"name": "svm_word_bigram_c1", "family": "linear_svm", "features": "word_bigram", "C": 1.0, "class_weight": "balanced"},
    {"name": "svm_word_bigram_c2", "family": "linear_svm", "features": "word_bigram", "C": 2.0, "class_weight": "balanced"},
    {"name": "svm_word_char_c1", "family": "linear_svm", "features": "word_char", "C": 1.0, "class_weight": "balanced"},
]


def load_partitions(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], MultiLabelBinarizer, dict[str, np.ndarray]]:
    frame = pd.read_csv(project_path(config["dataset"]["processed_path"]))
    splits = pd.read_csv(project_path(config["dataset"]["split_path"]))
    frame = frame.merge(splits, on="relationship_id", validate="one_to_one")
    partitions = {name: frame.loc[frame["split"] == name].reset_index(drop=True) for name in ("train", "validation", "test")}
    binarizer = MultiLabelBinarizer()
    binarizer.fit([labels.split("|") for labels in frame["tactics"]])
    targets = {
        name: binarizer.transform([labels.split("|") for labels in partition["tactics"]])
        for name, partition in partitions.items()
    }
    return partitions, binarizer, targets


def build_pipeline(experiment: dict[str, Any], config: dict[str, Any]) -> Pipeline:
    tfidf_config = config["tfidf"]
    features = make_tfidf(experiment["features"], tfidf_config["max_features"], tfidf_config["min_df"])
    if experiment["family"] == "logistic_regression":
        estimator = LogisticRegression(
            C=experiment["C"],
            class_weight=experiment["class_weight"],
            max_iter=config["logistic_regression"]["max_iter"],
            random_state=config["seed"],
        )
    else:
        estimator = LinearSVC(C=experiment["C"], class_weight=experiment["class_weight"], random_state=config["seed"])
    # Fifteen compact linear fits are quick enough sequentially and avoid large
    # sparse matrices being copied into worker processes on memory-limited hosts.
    return Pipeline([("features", features), ("classifier", OneVsRestClassifier(estimator, n_jobs=1))])


def _save_runtime_versions() -> None:
    packages = ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib", "PyYAML"]
    versions = {package: importlib.metadata.version(package) for package in packages}
    versions["python"] = platform.python_version()
    path = project_path("artifacts/metrics/package_versions.json")
    path.write_text(json.dumps(versions, indent=2), encoding="utf-8")


def _save_feature_weights(pipeline: Pipeline, labels: list[str], output_path: Path, top_n: int = 20) -> None:
    feature_names = pipeline.named_steps["features"].get_feature_names_out()
    estimators = pipeline.named_steps["classifier"].estimators_
    rows = []
    for label, estimator in zip(labels, estimators):
        top_indices = np.argsort(estimator.coef_[0])[-top_n:][::-1]
        rows.extend(
            {"tactic": label, "feature": feature_names[index], "weight": float(estimator.coef_[0, index]), "rank": rank}
            for rank, index in enumerate(top_indices, start=1)
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _misclassified_examples(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    wrong = np.any(y_true != y_pred, axis=1)
    rows = []
    for index in np.flatnonzero(wrong):
        true_labels = [labels[i] for i in np.flatnonzero(y_true[index])]
        predicted_labels = [labels[i] for i in np.flatnonzero(y_pred[index])]
        rows.append(
            {
                "text": frame.iloc[index]["text"],
                "true_labels": "|".join(true_labels),
                "predicted_labels": "|".join(predicted_labels),
                "max_confidence": float(probabilities[index].max()),
                "source_name": frame.iloc[index]["source_name"],
                "technique_id": frame.iloc[index]["technique_id"],
                "technique_name": frame.iloc[index]["technique_name"],
            }
        )
    return pd.DataFrame(rows).sort_values("max_confidence", ascending=False)


def train(config_path: str = "configs/experiment.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    np.random.seed(config["seed"])
    partitions, binarizer, targets = load_partitions(config)
    labels = binarizer.classes_.tolist()
    metrics_dir = project_path("artifacts/metrics")
    model_dir = project_path("artifacts/models")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    most_common_set = Counter(partitions["train"]["tactics"]).most_common(1)[0][0]
    baseline_row = binarizer.transform([most_common_set.split("|")])[0]
    baseline_predictions = np.tile(baseline_row, (len(partitions["validation"]), 1))
    baseline_metrics = multilabel_metrics(targets["validation"], baseline_predictions, labels)
    results = [{"name": "most_frequent_label_set", "family": "trivial_baseline", "features": "none", "C": None, **baseline_metrics}]

    trained: dict[str, Pipeline] = {}
    for experiment in EXPERIMENTS:
        LOGGER.info("Training %s", experiment["name"])
        started = time.perf_counter()
        pipeline = build_pipeline(experiment, config)
        pipeline.fit(partitions["train"]["text"], targets["train"])
        predictions = pipeline.predict(partitions["validation"]["text"])
        metrics = multilabel_metrics(targets["validation"], predictions, labels)
        results.append({**experiment, **metrics, "training_seconds": time.perf_counter() - started})
        trained[experiment["name"]] = pipeline

    results_frame = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_frame.drop(columns=["per_label_f1"]).to_csv(metrics_dir / "classical_experiments.csv", index=False)
    (metrics_dir / "classical_validation_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for family in ("logistic_regression", "linear_svm"):
        candidates = results_frame.loc[results_frame["family"] == family]
        best_name = str(candidates.iloc[0]["name"])
        best_pipeline = trained[best_name]
        test_predictions = best_pipeline.predict(partitions["test"]["text"])
        test_metrics = multilabel_metrics(targets["test"], test_predictions, labels)
        (metrics_dir / f"{family}_test_metrics.json").write_text(
            json.dumps({"selected_experiment": best_name, **test_metrics}, indent=2), encoding="utf-8"
        )
        joblib.dump({"pipeline": best_pipeline, "binarizer": binarizer, "config": config}, model_dir / f"{family}.joblib")
        if family == "logistic_regression":
            probabilities = best_pipeline.predict_proba(partitions["test"]["text"])
            errors = _misclassified_examples(partitions["test"], targets["test"], test_predictions, probabilities, labels)
            errors.to_csv(metrics_dir / "misclassified_examples.csv", index=False)
            _save_feature_weights(best_pipeline, labels, metrics_dir / "top_tfidf_features.csv")

    model_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_dir.glob("*.joblib")
    }
    (metrics_dir / "model_artifact_hashes.json").write_text(json.dumps(model_hashes, indent=2), encoding="utf-8")
    _save_runtime_versions()
    return results_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ATT&CK tactic classifiers")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = train(args.config)
    print(results[["name", "macro_f1", "micro_f1", "samples_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
