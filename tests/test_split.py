import pandas as pd

from src.data.split import exclude_cross_split_near_duplicates, make_grouped_splits


def _fixture_frame() -> pd.DataFrame:
    rows = []
    for group_number in range(20):
        for example in range(3):
            rows.append(
                {
                    "text": f"procedure {group_number} {example}",
                    "source_id": f"G{group_number:04d}",
                    "tactics": "execution|discovery" if example == 0 else "execution",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_has_no_source_or_text_overlap() -> None:
    frame = _fixture_frame()
    assignments, _ = make_grouped_splits(frame, seed=42, test_size=0.20, validation_size=0.15)
    partitions = {name: frame.loc[assignments == name] for name in assignments.unique()}

    assert set(partitions["train"].source_id).isdisjoint(partitions["test"].source_id)
    assert set(partitions["train"].source_id).isdisjoint(partitions["validation"].source_id)
    assert set(partitions["validation"].source_id).isdisjoint(partitions["test"].source_id)
    assert set(partitions["train"].text).isdisjoint(partitions["test"].text)


def test_grouped_split_is_deterministic() -> None:
    frame = _fixture_frame()
    first, _ = make_grouped_splits(frame, 42, 0.20, 0.15)
    second, _ = make_grouped_splits(frame, 42, 0.20, 0.15)

    assert first.tolist() == second.tolist()


def test_near_duplicate_purge_keeps_more_conservative_partition() -> None:
    frame = pd.DataFrame(
        {
            "relationship_id": ["train-copy", "test-copy", "unrelated"],
            "text": [
                "The actor executed a PowerShell payload on the compromised host",
                "The actor executed a PowerShell payload on the compromised host today",
                "Credentials were obtained from browser password stores",
            ],
        }
    )
    assignments = pd.Series(["train", "test", "validation"])

    cleaned, excluded = exclude_cross_split_near_duplicates(frame, assignments, threshold=0.75)

    assert cleaned.tolist() == ["excluded_near_duplicate", "test", "validation"]
    assert excluded == ["train-copy"]
