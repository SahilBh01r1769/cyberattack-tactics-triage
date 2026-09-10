import numpy as np

from src.features.text_features import make_tfidf
from src.models.train_classical import build_pipeline


def test_tfidf_pipeline_trains_on_tiny_fixture() -> None:
    config = {
        "seed": 42,
        "tfidf": {"max_features": 100, "min_df": 1},
        "logistic_regression": {"max_iter": 100},
    }
    experiment = {
        "family": "logistic_regression",
        "features": "word_bigram",
        "C": 1.0,
        "class_weight": None,
    }
    texts = ["runs powershell command", "collects browser files", "executes shell", "archives documents"]
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

    pipeline = build_pipeline(experiment, config)
    pipeline.fit(texts, targets)

    assert pipeline.predict(["powershell script"]).shape == (1, 2)
    assert pipeline.predict_proba(["powershell script"]).shape == (1, 2)


def test_unknown_feature_configuration_fails_clearly() -> None:
    try:
        make_tfidf("unknown")
    except ValueError as exc:
        assert "Unknown TF-IDF" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
