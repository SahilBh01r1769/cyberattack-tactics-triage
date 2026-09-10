from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import load_config, project_path
from src.models.calibrate import route_predictions, routing_confidence


@dataclass
class TacticPredictor:
    pipeline: Any
    calibration_coefficients: np.ndarray
    calibration_intercepts: np.ndarray
    labels: list[str]
    label_thresholds: np.ndarray
    routing_threshold: float

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path = "artifacts/models/confidence_logistic_regression.joblib",
        routing_threshold: float | None = None,
    ) -> "TacticPredictor":
        artifact = joblib.load(project_path(artifact_path))
        config = artifact.get("config") or load_config()
        return cls(
            pipeline=artifact["pipeline"],
            calibration_coefficients=np.asarray(artifact["calibration_coefficients"]),
            calibration_intercepts=np.asarray(artifact["calibration_intercepts"]),
            labels=list(artifact["labels"]),
            label_thresholds=np.asarray(artifact["label_thresholds"]),
            routing_threshold=routing_threshold if routing_threshold is not None else config["confidence"]["default_threshold"],
        )

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Threat description cannot be empty")
        scores = np.asarray(self.pipeline.decision_function([cleaned])).reshape(1, -1)
        logits = scores * self.calibration_coefficients + self.calibration_intercepts
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        predicted = (probabilities >= self.label_thresholds).astype(int)
        confidence = float(routing_confidence(probabilities, predicted)[0])
        decision = str(route_predictions(np.asarray([confidence]), self.routing_threshold)[0])
        selected = [
            {"tactic": label, "confidence": float(probabilities[0, index])}
            for index, label in enumerate(self.labels)
            if predicted[0, index]
        ]
        selected.sort(key=lambda item: item["confidence"], reverse=True)
        top_index = int(np.argmax(probabilities[0]))
        return {
            "predictions": selected,
            "top_candidate": {"tactic": self.labels[top_index], "confidence": float(probabilities[0, top_index])},
            "routing_confidence": confidence,
            "routing_threshold": self.routing_threshold,
            "decision": decision,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ATT&CK tactics for a short CTI description")
    parser.add_argument("--text", required=True, help="Threat or incident description")
    parser.add_argument("--model", default="artifacts/models/confidence_logistic_regression.joblib")
    parser.add_argument("--threshold", type=float, default=None, help="Override analyst-routing confidence threshold")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    predictor = TacticPredictor.from_artifact(args.model, args.threshold)
    result = predictor.predict(args.text)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("Predicted tactics:")
    if result["predictions"]:
        for prediction in result["predictions"]:
            print(f"  {prediction['tactic'].replace('-', ' ').title():24} {prediction['confidence']:.3f}")
    else:
        candidate = result["top_candidate"]
        print(f"  No tactic passed its label threshold (top: {candidate['tactic']}, {candidate['confidence']:.3f})")
    print(f"\nDecision: {result['decision'].replace('_', '-')}")
    print(f"Routing confidence: {result['routing_confidence']:.3f} (threshold {result['routing_threshold']:.2f})")


if __name__ == "__main__":
    main()
