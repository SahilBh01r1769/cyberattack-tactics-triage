import numpy as np

from src.evaluation.evaluate import label_confusion_counts, transformer_comparison_row


def test_label_confusion_counts() -> None:
    truth = np.array([[1, 0], [0, 1], [1, 1]])
    predicted = np.array([[1, 1], [0, 0], [0, 1]])

    result = label_confusion_counts(truth, predicted, ["execution", "collection"])

    execution = result.set_index("tactic").loc["execution"]
    assert execution["true_positive"] == 1
    assert execution["false_negative"] == 1
    assert execution["false_positive"] == 0


def test_smoke_transformer_result_is_not_added_to_comparison() -> None:
    assert transformer_comparison_row({"smoke_limit": 128, "validation_tuned_test_metrics": {"macro_f1": 0.1}}) is None


def test_full_transformer_result_can_be_added_to_comparison() -> None:
    row = transformer_comparison_row(
        {
            "model_name": "distilbert-base-uncased",
            "smoke_limit": None,
            "validation_tuned_test_metrics": {"macro_f1": 0.77},
        }
    )

    assert row == {"model": "DistilBERT (distilbert-base-uncased)", "macro_f1": 0.77}
