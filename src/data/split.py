from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import load_config, project_path


@dataclass(frozen=True)
class SplitCandidate:
    train_indices: np.ndarray
    holdout_indices: np.ndarray
    seed: int
    score: float


def parse_labels(series: pd.Series) -> list[list[str]]:
    return [value.split("|") for value in series]


def _distribution_score(all_targets: np.ndarray, holdout_targets: np.ndarray, actual_size: float, target_size: float) -> float:
    overall_prevalence = all_targets.mean(axis=0)
    holdout_prevalence = holdout_targets.mean(axis=0)
    return float(np.abs(overall_prevalence - holdout_prevalence).mean() + abs(actual_size - target_size))


def select_grouped_holdout(
    frame: pd.DataFrame,
    targets: np.ndarray,
    holdout_size: float,
    base_seed: int,
    attempts: int = 64,
) -> SplitCandidate:
    """Choose a strict group holdout with reasonable size and label balance."""
    best: SplitCandidate | None = None
    for offset in range(attempts):
        seed = base_seed + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
        train_indices, holdout_indices = next(splitter.split(frame, groups=frame["source_id"]))
        if (targets[holdout_indices].sum(axis=0) == 0).any():
            continue
        score = _distribution_score(targets, targets[holdout_indices], len(holdout_indices) / len(frame), holdout_size)
        candidate = SplitCandidate(train_indices, holdout_indices, seed, score)
        if best is None or candidate.score < best.score:
            best = candidate
    if best is None:
        raise ValueError("Could not find a grouped split containing every tactic label")
    return best


def make_grouped_splits(frame: pd.DataFrame, seed: int, test_size: float, validation_size: float) -> tuple[pd.Series, dict]:
    labels = parse_labels(frame["tactics"])
    targets = MultiLabelBinarizer().fit_transform(labels)

    test_candidate = select_grouped_holdout(frame, targets, test_size, seed)
    remaining = frame.iloc[test_candidate.train_indices].reset_index().rename(columns={"index": "original_index"})
    remaining_targets = targets[test_candidate.train_indices]
    relative_validation_size = validation_size / (1.0 - test_size)
    validation_candidate = select_grouped_holdout(
        remaining,
        remaining_targets,
        relative_validation_size,
        seed + 1000,
    )

    assignments = pd.Series("train", index=frame.index, name="split")
    assignments.iloc[test_candidate.holdout_indices] = "test"
    validation_original_indices = remaining.iloc[validation_candidate.holdout_indices]["original_index"].to_numpy()
    assignments.iloc[validation_original_indices] = "validation"

    metadata = {
        "strategy": "grouped_source",
        "group_column": "source_id",
        "test_selection_seed": test_candidate.seed,
        "validation_selection_seed": validation_candidate.seed,
        "candidate_attempts": 64,
        "test_balance_score": test_candidate.score,
        "validation_balance_score": validation_candidate.score,
    }
    return assignments, metadata


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return dict(sorted(Counter(label for labels in parse_labels(frame["tactics"]) for label in labels).items()))


def cross_split_near_duplicates(frame: pd.DataFrame, assignments: pd.Series, threshold: float) -> dict:
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=75000).fit_transform(frame["text"])
    distances, indices = NearestNeighbors(n_neighbors=4, metric="cosine").fit(matrix).kneighbors(matrix)
    pairs = {}
    for row in range(len(frame)):
        for neighbor_position in range(1, indices.shape[1]):
            neighbor = int(indices[row, neighbor_position])
            similarity = 1.0 - float(distances[row, neighbor_position])
            pair = tuple(sorted((row, neighbor)))
            if similarity >= threshold and assignments.iloc[row] != assignments.iloc[neighbor]:
                pairs[pair] = max(similarity, pairs.get(pair, 0.0))
    examples = [
        {
            "left_relationship_id": frame.iloc[left]["relationship_id"],
            "left_split": assignments.iloc[left],
            "right_relationship_id": frame.iloc[right]["relationship_id"],
            "right_split": assignments.iloc[right],
            "cosine_similarity": similarity,
        }
        for (left, right), similarity in sorted(pairs.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    return {"threshold": threshold, "pair_count": len(pairs), "examples": examples}


def split_diagnostics(frame: pd.DataFrame, assignments: pd.Series, metadata: dict, seed: int, near_duplicate_threshold: float) -> dict:
    partitions = {name: frame.loc[assignments == name] for name in ("train", "validation", "test")}
    train_random, test_random = train_test_split(frame, test_size=0.20, random_state=seed)
    exact_overlap = {
        "train_validation": len(set(partitions["train"].text) & set(partitions["validation"].text)),
        "train_test": len(set(partitions["train"].text) & set(partitions["test"].text)),
        "validation_test": len(set(partitions["validation"].text) & set(partitions["test"].text)),
    }
    source_overlap = {
        "train_validation": len(set(partitions["train"].source_id) & set(partitions["validation"].source_id)),
        "train_test": len(set(partitions["train"].source_id) & set(partitions["test"].source_id)),
        "validation_test": len(set(partitions["validation"].source_id) & set(partitions["test"].source_id)),
    }
    return {
        **metadata,
        "counts": {name: len(partition) for name, partition in partitions.items()},
        "unique_sources": {name: partition["source_id"].nunique() for name, partition in partitions.items()},
        "label_counts": {name: _label_counts(partition) for name, partition in partitions.items()},
        "exact_text_overlap": exact_overlap,
        "source_overlap": source_overlap,
        "random_split_source_overlap": len(set(train_random.source_id) & set(test_random.source_id)),
        "technique_overlap_train_test": len(set(partitions["train"].technique_id) & set(partitions["test"].technique_id)),
        "cross_split_near_duplicates": cross_split_near_duplicates(frame, assignments, near_duplicate_threshold),
    }


def build_splits(config_path: str = "configs/experiment.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    frame = pd.read_csv(project_path(config["dataset"]["processed_path"]))
    split_config = config["split"]
    assignments, metadata = make_grouped_splits(
        frame,
        config["seed"],
        split_config["test_size"],
        split_config["validation_size"],
    )
    split_frame = frame[["relationship_id"]].copy()
    split_frame["split"] = assignments
    output_path = project_path(config["dataset"]["split_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(output_path, index=False)

    diagnostics = split_diagnostics(
        frame,
        assignments,
        metadata,
        config["seed"],
        config["dataset"]["near_duplicate_threshold"],
    )
    diagnostics_path = project_path("artifacts/metrics/split_diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return split_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible ATT&CK dataset splits")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    result = build_splits(args.config)
    print(result["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
