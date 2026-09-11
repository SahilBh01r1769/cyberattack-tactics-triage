from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import multilabel_metrics
from src.models.calibrate import choose_label_thresholds


LOGGER = logging.getLogger(__name__)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _prediction_frame(
    frame: pd.DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    rows = []
    for index in range(len(frame)):
        rows.append(
            {
                "text": frame.iloc[index]["text"],
                "true_labels": "|".join(labels[column] for column in np.flatnonzero(targets[index])),
                "predicted_labels": "|".join(labels[column] for column in np.flatnonzero(predictions[index])),
                "top_confidence": float(probabilities[index].max()),
                "exact_match": bool(np.array_equal(targets[index], predictions[index])),
            }
        )
    return pd.DataFrame(rows)


def _package_evidence(paths: list[Path]) -> Path:
    archive_path = project_path("artifacts/transformer_evidence.zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    return archive_path


def _require_transformer_dependencies():
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install transformer dependencies with: pip install -e '.[transformer]'") from exc
    return torch, Dataset, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments


def load_transformer_data(config: dict, smoke_limit: int | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frame = pd.read_csv(project_path(config["dataset"]["processed_path"]))
    assignments = pd.read_csv(project_path(config["dataset"]["split_path"]))
    frame = frame.merge(assignments, on="relationship_id", validate="one_to_one")
    labels = sorted({label for value in frame["tactics"] for label in value.split("|")})
    frame["labels"] = frame["tactics"].map(
        lambda value: [1.0 if label in value.split("|") else 0.0 for label in labels]
    )
    partitions = {}
    for split in ("train", "validation", "test"):
        partition = frame.loc[frame["split"] == split, ["text", "labels"]].reset_index(drop=True)
        if smoke_limit:
            limit = smoke_limit if split == "train" else max(32, smoke_limit // 4)
            partition = partition.sample(min(limit, len(partition)), random_state=config["seed"]).reset_index(drop=True)
        partitions[split] = partition
    return partitions, labels


def train(config_path: str = "configs/experiment.yaml", smoke_limit: int | None = None) -> dict:
    (
        torch,
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    ) = _require_transformer_dependencies()
    config = load_config(config_path)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    partitions, labels = load_transformer_data(config, smoke_limit)
    transformer_config = config["transformer"]
    model_name = transformer_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    datasets = {}
    for name, frame in partitions.items():
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        datasets[name] = dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=transformer_config["max_length"]),
            batched=True,
            remove_columns=["text"],
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
        problem_type="multi_label_classification",
    )
    train_targets = np.asarray(partitions["train"]["labels"].tolist())
    positives = train_targets.sum(axis=0)
    pos_weights = torch.tensor((len(train_targets) - positives) / np.maximum(positives, 1), dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels_tensor = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(outputs.logits.device))(
                outputs.logits, labels_tensor.float()
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(prediction_output):
        logits, target_values = prediction_output
        probabilities = _sigmoid(logits)
        predictions = (probabilities >= 0.5).astype(int)
        metrics = multilabel_metrics(target_values.astype(int), predictions, labels)
        return {key: value for key, value in metrics.items() if key != "per_label_f1"}

    output_dir = project_path("artifacts/transformer_checkpoints")
    epochs = 1 if smoke_limit else transformer_config["epochs"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=transformer_config["learning_rate"],
        per_device_train_batch_size=transformer_config["batch_size"],
        per_device_eval_batch_size=transformer_config["batch_size"],
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        report_to=[],
        fp16=torch.cuda.is_available(),
        logging_steps=max(1, len(datasets["train"]) // (transformer_config["batch_size"] * 10)),
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    train_result = trainer.train()
    validation_result = trainer.predict(datasets["validation"])
    test_result = trainer.predict(datasets["test"])
    validation_probabilities = _sigmoid(validation_result.predictions)
    test_probabilities = _sigmoid(test_result.predictions)
    label_thresholds = choose_label_thresholds(validation_result.label_ids.astype(int), validation_probabilities)
    fixed_predictions = (test_probabilities >= 0.5).astype(int)
    tuned_predictions = (test_probabilities >= label_thresholds).astype(int)
    fixed_test_metrics = multilabel_metrics(test_result.label_ids.astype(int), fixed_predictions, labels)
    tuned_test_metrics = multilabel_metrics(test_result.label_ids.astype(int), tuned_predictions, labels)

    metrics = {
        "model_name": model_name,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "smoke_limit": smoke_limit,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "training_metrics": train_result.metrics,
        "label_thresholds_selected_on_validation": dict(zip(labels, label_thresholds.tolist())),
        "fixed_0_5_test_metrics": fixed_test_metrics,
        "validation_tuned_test_metrics": tuned_test_metrics,
        "log_history": trainer.state.log_history,
    }
    metrics_path = project_path("artifacts/metrics/transformer_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_path = project_path("artifacts/metrics/transformer_test_predictions.csv")
    _prediction_frame(
        partitions["test"],
        test_result.label_ids.astype(int),
        tuned_predictions,
        test_probabilities,
        labels,
    ).to_csv(predictions_path, index=False)
    versions_path = project_path("artifacts/metrics/transformer_package_versions.json")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "datasets", "accelerate", "numpy", "pandas")
    }
    versions_path.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    status_path = project_path("artifacts/metrics/transformer_run_status.json")
    status_path.write_text(
        json.dumps({"status": "completed", "model_name": model_name, "device": metrics["device"], "smoke_limit": smoke_limit}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")
    evidence_path = _package_evidence(
        [metrics_path, predictions_path, versions_path, status_path, project_path("configs/experiment.yaml")]
    )
    metrics["evidence_archive"] = str(evidence_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for ATT&CK tactic classification")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--smoke-limit", type=int, default=None, help="Bound training examples for a dependency/flow smoke run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(train(args.config, args.smoke_limit), indent=2))


if __name__ == "__main__":
    main()
