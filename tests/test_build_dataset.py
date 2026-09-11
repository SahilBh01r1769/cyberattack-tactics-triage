import json
from pathlib import Path

import pandas as pd

from src.data.build_dataset import extract_records, extract_tactic_catalog, normalize_text, remove_duplicate_text


FIXTURE = Path(__file__).parent / "fixtures" / "attack_bundle.json"


def test_parser_extracts_procedure_with_provenance() -> None:
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    records = extract_records(bundle)

    assert len(records) == 2
    assert records[0]["technique_id"] == "T1059.001"
    assert records[0]["tactics"] == "defense-evasion|execution"
    assert "invented-tactic" not in records[0]["tactics"]
    assert records[0]["source_id"] == "G0001"
    assert "<b>" not in records[0]["text"]


def test_duplicate_text_removal_is_case_insensitive() -> None:
    frame = pd.DataFrame({"text": ["Same procedure text", "same procedure text", "Different text"]})
    deduplicated, removed = remove_duplicate_text(frame)

    assert removed == 1
    assert deduplicated["text"].tolist() == ["Same procedure text", "Different text"]


def test_normalization_keeps_link_text_and_removes_citation_markup() -> None:
    text = "[PowerShell](https://attack.mitre.org/software/S0194) ran commands.(Citation: Example Report 2024)"

    assert normalize_text(text) == "PowerShell ran commands."


def test_tactic_catalog_uses_stix_metadata() -> None:
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    catalog = extract_tactic_catalog(bundle)

    assert [tactic["slug"] for tactic in catalog] == ["execution", "defense-evasion"]
    assert catalog[0]["name"] == "Execution"
