from src.config import load_config
from src.models.train_transformer import load_transformer_data


def test_transformer_data_uses_shared_splits_and_multihot_labels() -> None:
    partitions, labels = load_transformer_data(load_config(), smoke_limit=40)

    assert len(partitions["train"]) == 40
    assert len(labels) == 15
    assert all(len(values) == len(labels) for values in partitions["train"]["labels"])
    assert set(partitions) == {"train", "validation", "test"}
