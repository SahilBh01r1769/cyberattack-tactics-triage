import numpy as np

from src.evaluation.evaluate import label_confusion_counts


def test_label_confusion_counts() -> None:
    truth = np.array([[1, 0], [0, 1], [1, 1]])
    predicted = np.array([[1, 1], [0, 0], [0, 1]])

    result = label_confusion_counts(truth, predicted, ["execution", "collection"])

    execution = result.set_index("tactic").loc["execution"]
    assert execution["true_positive"] == 1
    assert execution["false_negative"] == 1
    assert execution["false_positive"] == 0
