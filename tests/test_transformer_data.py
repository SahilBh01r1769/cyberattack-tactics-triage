import pandas as pd

from src.models.train_transformer import load_transformer_data


def test_transformer_data_uses_shared_splits_and_multihot_labels(tmp_path) -> None:
    records = []
    assignments = []
    split_names = ["train"] * 8 + ["validation"] * 4 + ["test"] * 4
    for index, split in enumerate(split_names):
        relationship_id = f"relationship--{index}"
        records.append(
            {
                "relationship_id": relationship_id,
                "text": f"Procedure example {index}",
                "tactics": "execution|discovery" if index % 3 == 0 else "execution",
            }
        )
        assignments.append({"relationship_id": relationship_id, "split": split})
    dataset_path = tmp_path / "dataset.csv"
    split_path = tmp_path / "splits.csv"
    pd.DataFrame(records).to_csv(dataset_path, index=False)
    pd.DataFrame(assignments).to_csv(split_path, index=False)
    config = {"seed": 42, "dataset": {"processed_path": str(dataset_path), "split_path": str(split_path)}}

    partitions, labels = load_transformer_data(config)

    assert len(partitions["train"]) == 8
    assert labels == ["discovery", "execution"]
    assert all(len(values) == len(labels) for values in partitions["train"]["labels"])
    assert set(partitions) == {"train", "validation", "test"}
