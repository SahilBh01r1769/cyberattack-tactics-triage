from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from src.config import project_path
from src.inference.predict import TacticPredictor


VALIDATION_TEXTS = [
    "The actor used PowerShell to execute a payload.",
    "Credentials were captured through a keylogger.",
    "The malware archived documents before exfiltration.",
]


def package_artifact(source: Path, destination: Path) -> dict[str, object]:
    """Create a compressed, inference-only artifact and verify output parity."""
    source_artifact = joblib.load(source)
    packaged = {
        key: source_artifact[key]
        for key in (
            "pipeline",
            "config",
            "calibration_coefficients",
            "calibration_intercepts",
            "label_thresholds",
            "labels",
        )
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(packaged, destination, compress=("xz", 3))

    source_predictions = TacticPredictor.from_artifact(source).predict_many(VALIDATION_TEXTS)
    packaged_predictions = TacticPredictor.from_artifact(destination).predict_many(VALIDATION_TEXTS)
    if source_predictions != packaged_predictions:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Packaged model predictions differ from the source artifact")

    model_bytes = destination.read_bytes()
    return {
        "path": str(destination.relative_to(project_path("."))),
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "size_bytes": len(model_bytes),
        "source_path": str(source.relative_to(project_path("."))),
        "parity_examples": len(VALIDATION_TEXTS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the calibrated model for the Streamlit demo")
    parser.add_argument("--source", default="artifacts/models/confidence_logistic_regression.joblib")
    parser.add_argument("--output", default="artifacts/models/triage_lr.joblib")
    parser.add_argument("--metadata", default="artifacts/metrics/deployable_model.json")
    args = parser.parse_args()

    metadata = package_artifact(project_path(args.source), project_path(args.output))
    metadata_path = project_path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
