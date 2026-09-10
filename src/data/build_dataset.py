from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import load_config, project_path


LOGGER = logging.getLogger(__name__)
TAG_PATTERN = re.compile(r"<[^>]+>")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^]]+)]\([^)]+\)")
CITATION_PATTERN = re.compile(r"\s*\(Citation:\s*[^)]+\)")
SPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Remove markup artifacts while retaining the wording of a procedure example."""
    value = html.unescape(value)
    value = TAG_PATTERN.sub(" ", value)
    value = MARKDOWN_LINK_PATTERN.sub(r"\1", value)
    value = CITATION_PATTERN.sub("", value)
    return SPACE_PATTERN.sub(" ", value).strip()


def _external_id(obj: dict[str, Any]) -> str:
    references = obj.get("external_references", [])
    attack_ref = next((ref for ref in references if ref.get("source_name") == "mitre-attack"), None)
    return (attack_ref or {}).get("external_id", "")


def _citation_sources(relationship: dict[str, Any]) -> str:
    sources = {
        str(ref.get("source_name", "")).strip()
        for ref in relationship.get("external_references", [])
        if ref.get("source_name")
    }
    return "|".join(sorted(sources))


def extract_records(bundle: dict[str, Any], min_text_chars: int = 30) -> list[dict[str, Any]]:
    """Extract technique-linked procedure descriptions from an ATT&CK STIX bundle."""
    objects = bundle.get("objects", [])
    active = {
        obj["id"]: obj
        for obj in objects
        if "id" in obj and not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)
    }
    valid_tactics = {
        obj.get("x_mitre_shortname")
        for obj in active.values()
        if obj.get("type") == "x-mitre-tactic" and obj.get("x_mitre_shortname")
    }
    records: list[dict[str, Any]] = []

    for relationship in objects:
        if relationship.get("type") != "relationship" or relationship.get("relationship_type") != "uses":
            continue
        source = active.get(relationship.get("source_ref", ""))
        technique = active.get(relationship.get("target_ref", ""))
        if not source or not technique or technique.get("type") != "attack-pattern":
            continue

        text = normalize_text(relationship.get("description", ""))
        tactics = sorted(
            {
                phase.get("phase_name", "").strip()
                for phase in technique.get("kill_chain_phases", [])
                if phase.get("kill_chain_name") == "mitre-attack"
                and phase.get("phase_name") in valid_tactics
            }
        )
        if len(text) < min_text_chars or not tactics:
            continue

        records.append(
            {
                "text": text,
                "technique_id": _external_id(technique),
                "technique_name": technique.get("name", ""),
                "tactics": "|".join(tactics),
                "source_type": source.get("type", ""),
                "source_id": _external_id(source) or source.get("id", ""),
                "source_name": source.get("name", ""),
                "citations": _citation_sources(relationship),
                "relationship_id": relationship.get("id", ""),
            }
        )
    return records


def remove_duplicate_text(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact normalized-text duplicates, preferring the first STIX relationship."""
    if frame.empty:
        return frame.copy(), 0
    keys = frame["text"].str.casefold().str.strip()
    duplicate_count = int(keys.duplicated().sum())
    return frame.loc[~keys.duplicated()].reset_index(drop=True), duplicate_count


def estimate_near_duplicates(texts: pd.Series, threshold: float) -> dict[str, Any]:
    """Estimate near-duplicate pairs using word n-gram cosine similarity."""
    if len(texts) < 2:
        return {"threshold": threshold, "pair_count": 0, "affected_samples": 0}
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=75000).fit_transform(texts)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine").fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    pairs = {
        tuple(sorted((row, int(indices[row, 1]))))
        for row in range(len(texts))
        if 1.0 - float(distances[row, 1]) >= threshold
    }
    return {
        "threshold": threshold,
        "pair_count": len(pairs),
        "affected_samples": len({index for pair in pairs for index in pair}),
    }


def dataset_statistics(
    frame: pd.DataFrame,
    exact_duplicates_removed: int,
    near_duplicate_threshold: float,
    source_sha256: str,
) -> dict[str, Any]:
    labels = frame["tactics"].str.split("|")
    label_counts = Counter(label for row in labels for label in row)
    text_lengths = frame["text"].str.split().str.len()
    near_duplicates = estimate_near_duplicates(frame["text"], near_duplicate_threshold)
    return {
        "usable_samples": int(len(frame)),
        "techniques": int(frame["technique_id"].nunique()),
        "tactics": len(label_counts),
        "label_distribution": dict(sorted(label_counts.items())),
        "multilabel_samples": int(labels.map(len).gt(1).sum()),
        "multilabel_fraction": float(labels.map(len).gt(1).mean()),
        "text_length_words": {
            "min": int(text_lengths.min()),
            "median": float(text_lengths.median()),
            "mean": float(text_lengths.mean()),
            "p95": float(np.percentile(text_lengths, 95)),
            "max": int(text_lengths.max()),
        },
        "exact_duplicates_removed": exact_duplicates_removed,
        "near_duplicates": near_duplicates,
        "top_sources": frame["source_name"].value_counts().head(15).to_dict(),
        "source_type_distribution": frame["source_type"].value_counts().to_dict(),
        "source_stix_sha256": source_sha256,
    }


def download_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading Enterprise ATT&CK STIX from %s", url)
    urllib.request.urlretrieve(url, destination)


def build_dataset(config_path: str = "configs/experiment.yaml", force_download: bool = False) -> pd.DataFrame:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    raw_path = project_path(dataset_config["raw_path"])
    if force_download and raw_path.exists():
        raw_path.unlink()
    download_if_missing(dataset_config["stix_url"], raw_path)

    raw_bytes = raw_path.read_bytes()
    bundle = json.loads(raw_bytes)
    frame = pd.DataFrame(extract_records(bundle, dataset_config["min_text_chars"]))
    frame, duplicates_removed = remove_duplicate_text(frame)
    frame = frame.sort_values(["technique_id", "source_id", "relationship_id"]).reset_index(drop=True)

    output_path = project_path(dataset_config["processed_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    stats = dataset_statistics(
        frame,
        duplicates_removed,
        dataset_config["near_duplicate_threshold"],
        hashlib.sha256(raw_bytes).hexdigest(),
    )
    stats_path = project_path(dataset_config["statistics_path"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d records to %s", len(frame), output_path)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ATT&CK tactic dataset")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_dataset(args.config, args.force_download)


if __name__ == "__main__":
    main()
