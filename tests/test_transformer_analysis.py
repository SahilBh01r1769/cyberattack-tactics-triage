import pandas as pd

from src.evaluation.transformer_analysis import summarize_prediction_errors


def test_transformer_error_summary_counts_partial_multilabel_error() -> None:
    predictions = pd.DataFrame(
        {
            "text": ["one", "two"],
            "true_labels": ["execution", "collection|discovery"],
            "predicted_labels": ["execution", "discovery"],
            "top_confidence": [0.9, 0.8],
            "exact_match": [True, False],
        }
    )

    summary, errors = summarize_prediction_errors(predictions, ["collection", "discovery", "execution"])

    assert summary["examples_with_any_label_error"] == 1
    assert summary["most_missed_labels"] == {"collection": 1}
    assert len(errors) == 1
