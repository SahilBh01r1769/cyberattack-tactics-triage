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
        artifact_path: str | Path = "artifacts/models/triage_lr.joblib",
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

    def predict_many(
        self,
        texts: list[str],
        routing_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Classify a batch in one vectorized model call."""
        cleaned = [text.strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Threat descriptions cannot be empty")

        scores = np.asarray(self.pipeline.decision_function(cleaned))
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        logits = scores * self.calibration_coefficients + self.calibration_intercepts
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        predicted = (probabilities >= self.label_thresholds).astype(int)
        confidences = routing_confidence(probabilities, predicted)
        threshold = self.routing_threshold if routing_threshold is None else routing_threshold
        decisions = route_predictions(confidences, threshold)

        results = []
        for row in range(len(cleaned)):
            selected = [
                {"tactic": label, "confidence": float(probabilities[row, index])}
                for index, label in enumerate(self.labels)
                if predicted[row, index]
            ]
            selected.sort(key=lambda item: item["confidence"], reverse=True)
            top_index = int(np.argmax(probabilities[row]))
            results.append(
                {
                    "predictions": selected,
                    "top_candidate": {
                        "tactic": self.labels[top_index],
                        "confidence": float(probabilities[row, top_index]),
                    },
                    "routing_confidence": float(confidences[row]),
                    "routing_threshold": float(threshold),
                    "decision": str(decisions[row]),
                }
            )
        return results

    def predict(self, text: str, routing_threshold: float | None = None) -> dict[str, Any]:
        return self.predict_many([text], routing_threshold)[0]

    def explain(self, text: str, tactic: str, top_n: int = 6) -> list[dict[str, Any]]:
        """Return the strongest positive TF-IDF contributions for one tactic."""
        if tactic not in self.labels:
            raise ValueError(f"Unknown tactic: {tactic}")
        features = self.pipeline.named_steps.get("features")
        classifier = self.pipeline.named_steps.get("classifier")
        if features is None or classifier is None or not hasattr(classifier, "estimators_"):
            return []

        row = features.transform([text.strip()])
        label_index = self.labels.index(tactic)
        estimator = classifier.estimators_[label_index]
        contributions = row.multiply(estimator.coef_[0]).tocoo()
        names = features.get_feature_names_out()
        ranked = sorted(
            (
                (str(names[column]).removeprefix("word__"), float(value))
                for column, value in zip(contributions.col, contributions.data, strict=True)
                if value > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [{"feature": feature, "contribution": value} for feature, value in ranked[:top_n]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ATT&CK tactics for a short CTI description")
    parser.add_argument("--text", required=True, help="Threat or incident description")
    parser.add_argument("--model", default="artifacts/models/triage_lr.joblib")
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
