import numpy as np

from src.models.calibrate import route_predictions, routing_confidence


def test_confidence_threshold_changes_routing_decision() -> None:
    probabilities = np.array([[0.82, 0.20], [0.58, 0.30]])
    predictions = np.array([[1, 0], [1, 0]])
    confidence = routing_confidence(probabilities, predictions)

    assert route_predictions(confidence, 0.70).tolist() == ["auto_route", "analyst_review"]


def test_routing_confidence_handles_no_positive_label() -> None:
    probabilities = np.array([[0.35, 0.20]])
    predictions = np.array([[0, 0]])

    assert routing_confidence(probabilities, predictions).tolist() == [0.35]
