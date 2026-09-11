import pandas as pd

from src.data.split import select_grouped_holdout


def test_grouped_holdout_accepts_technique_groups() -> None:
    frame = pd.DataFrame(
        {
            "technique_id": [f"T{group:04d}" for group in range(20) for _ in range(2)],
            "source_id": [f"G{row:04d}" for row in range(40)],
        }
    )
    targets = pd.DataFrame(
        {"execution": [1] * 40, "discovery": [row % 2 for row in range(40)]}
    ).to_numpy()

    split = select_grouped_holdout(frame, targets, 0.20, 42, group_column="technique_id")
    train_techniques = set(frame.iloc[split.train_indices]["technique_id"])
    test_techniques = set(frame.iloc[split.holdout_indices]["technique_id"])

    assert train_techniques.isdisjoint(test_techniques)
