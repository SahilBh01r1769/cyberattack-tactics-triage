import numpy as np
import pandas as pd

from src.data.split import select_grouped_holdout
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


def test_calibration_subsets_can_be_group_disjoint() -> None:
    frame = pd.DataFrame({"source_id": [f"G{group}" for group in range(20) for _ in range(2)]})
    targets = np.column_stack([np.ones(40), np.tile([0, 1], 20)])

    split = select_grouped_holdout(frame, targets, 0.5, 3042, attempts=64)

    calibration_sources = set(frame.iloc[split.train_indices]["source_id"])
    threshold_sources = set(frame.iloc[split.holdout_indices]["source_id"])
    assert calibration_sources.isdisjoint(threshold_sources)
