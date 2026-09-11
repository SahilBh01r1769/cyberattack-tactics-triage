from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import project_path
from src.inference.predict import TacticPredictor


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def format_tactic(slug: str) -> str:
    return slug.replace("-", " ").title().replace("And Control", "and Control")


def classify_frame(
    frame: pd.DataFrame,
    text_column: str,
    predictor: TacticPredictor,
    routing_threshold: float,
) -> pd.DataFrame:
    """Append compact, human-readable model outputs to an uploaded queue."""
    if text_column not in frame.columns:
        raise ValueError(f"Missing text column: {text_column}")
    texts = frame[text_column].fillna("").astype(str).str.strip()
    if texts.eq("").any():
        raise ValueError("Remove empty threat descriptions before classification")

    results = predictor.predict_many(texts.tolist(), routing_threshold)
    output = frame.copy()
    output["predicted_tactics"] = [
        " | ".join(format_tactic(item["tactic"]) for item in result["predictions"])
        or format_tactic(result["top_candidate"]["tactic"])
        for result in results
    ]
    output["routing_confidence"] = [round(result["routing_confidence"], 6) for result in results]
    output["decision"] = [result["decision"] for result in results]
    return output


def safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Prevent spreadsheet formula execution in downloaded user-controlled text."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]):
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value
        )
    return safe.to_csv(index=False).encode("utf-8")
