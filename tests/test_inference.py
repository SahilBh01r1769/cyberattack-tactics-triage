import numpy as np

from src.inference.predict import TacticPredictor


class FakePipeline:
    def decision_function(self, texts):
        return np.array([[2.0, -1.0]])


def test_inference_returns_expected_schema() -> None:
    predictor = TacticPredictor(
        pipeline=FakePipeline(),
        calibration_coefficients=np.array([1.0, 1.0]),
        calibration_intercepts=np.array([0.0, 0.0]),
        labels=["execution", "collection"],
        label_thresholds=np.array([0.5, 0.5]),
        routing_threshold=0.7,
    )

    result = predictor.predict("The actor ran a PowerShell command")

    assert result["decision"] == "auto_route"
    assert result["predictions"][0]["tactic"] == "execution"
    assert round(result["predictions"][0]["confidence"], 3) == 0.881
    assert set(result) == {"predictions", "top_candidate", "routing_confidence", "routing_threshold", "decision"}


def test_inference_rejects_empty_text() -> None:
    predictor = TacticPredictor(
        FakePipeline(), np.ones(2), np.zeros(2), ["execution", "collection"], np.array([0.5, 0.5]), 0.7
    )
    try:
        predictor.predict("  ")
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
