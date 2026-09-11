import hashlib
import json
from pathlib import Path

from src.inference.predict import TacticPredictor


MODEL = Path("artifacts/models/triage_lr.joblib")
METADATA = Path("artifacts/metrics/deployable_model.json")


def test_deployable_model_hash_and_inference() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == metadata["sha256"]
    result = TacticPredictor.from_artifact(MODEL).predict("The adversary executed PowerShell commands.")
    assert result["top_candidate"]["tactic"] == "execution"
    assert result["decision"] == "auto_route"
