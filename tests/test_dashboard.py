import numpy as np
import pandas as pd

from src.dashboard import classify_frame, format_tactic, safe_csv_bytes
from src.inference.predict import TacticPredictor


class FakePipeline:
    def decision_function(self, texts):
        return np.tile(np.array([[2.0, -1.0]]), (len(texts), 1))


def predictor() -> TacticPredictor:
    return TacticPredictor(
        FakePipeline(), np.ones(2), np.zeros(2), ["execution", "command-and-control"], np.array([0.5, 0.5]), 0.7
    )


def test_classify_frame_appends_routing_fields() -> None:
    result = classify_frame(pd.DataFrame({"description": ["A command ran", "Another command ran"]}), "description", predictor(), 0.95)

    assert result["predicted_tactics"].tolist() == ["Execution", "Execution"]
    assert result["decision"].tolist() == ["analyst_review", "analyst_review"]


def test_classify_frame_rejects_empty_rows() -> None:
    try:
        classify_frame(pd.DataFrame({"text": [""]}), "text", predictor(), 0.7)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty input to be rejected")


def test_csv_export_escapes_formulas() -> None:
    exported = safe_csv_bytes(pd.DataFrame({"text": ["=2+2", "ordinary"]})).decode()

    assert "'=2+2" in exported
    assert format_tactic("command-and-control") == "Command and Control"
