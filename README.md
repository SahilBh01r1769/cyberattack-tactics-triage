# Cyber Threat Intelligence ATT&CK Triage

[![tests](https://github.com/SahilBh01r1769/cyber/actions/workflows/tests.yml/badge.svg)](https://github.com/SahilBh01r1769/cyber/actions/workflows/tests.yml)

An NLP project that maps short cyber-threat descriptions to one or more Enterprise MITRE ATT&CK tactics. It trains on official ATT&CK procedure examples—no LLM or external classification API is used.

The project is both an ML investigation and a small triage aid. It is not intended to replace analyst judgment.

## Try the workbench

The Streamlit workbench supports individual threat triage, confidence-aware analyst routing, local feature explanations, CSV batch review and direct access to the experiment evidence.

```bash
git clone https://github.com/SahilBh01r1769/cyber.git
cd cyber
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-app.txt
streamlit run app.py
```

The repository includes a compressed, checksum-tested classical model, so the interface runs without retraining.

CLI inference is also available:

```bash
python -m src.inference.predict --text "The adversary executed a PowerShell command."
```

## Results

Results use a source-grouped holdout: an actor, malware family, tool or campaign cannot appear in more than one partition. Eight cross-partition near-duplicate records are explicitly excluded.

| Model | Macro F1 | Micro F1 | Samples F1 | Exact match |
|---|---:|---:|---:|---:|
| Most-frequent label set | 0.022 | 0.183 | 0.195 | 0.186 |
| TF-IDF + Logistic Regression | 0.742 | 0.802 | **0.809** | 0.663 |
| Calibrated TF-IDF + LR | 0.752 | 0.807 | 0.789 | 0.685 |
| TF-IDF + Linear SVM | **0.773** | **0.826** | 0.807 | **0.720** |

Linear SVM is the strongest verified model on the cleaned split. A previous DistilBERT run reached 0.779 macro F1, but it predates the near-duplicate cleanup and is kept as historical evidence until rerun rather than mixed into the current comparison.

![Grouped-source model comparison](artifacts/figures/model_comparison.png)

### Confidence-aware routing

Platt calibrators and label thresholds are now fitted on separate, source-disjoint halves of the validation partition. Calibration reduces macro Brier score from 0.0266 to 0.0213 and expected calibration error from 0.0416 to 0.0085.

| Routing threshold | Auto-route coverage | Micro F1 on accepted cases | Samples F1 on accepted cases |
|---:|---:|---:|---:|
| 0.50 | 86.8% | 0.854 | 0.870 |
| 0.70 | 78.0% | 0.878 | 0.897 |
| 0.90 | 59.4% | 0.913 | 0.928 |

### Unseen-technique stress test

A secondary test trains the selected SVM on 452 techniques and evaluates it on 114 entirely unseen techniques. Macro F1 falls to **0.516** with zero technique overlap. This is intentionally not presented as a competing benchmark: it shows that unfamiliar behaviors remain substantially harder than unfamiliar actors or tools.

## Data and methodology

The dataset builder reads MITRE's official [Enterprise ATT&CK STIX data](https://github.com/mitre-attack/attack-stix-data), resolves procedure-to-technique relationships and technique-to-tactic phases, cleans markup, removes exact duplicates and retains source provenance.

- 16,955 usable procedure examples, 611 techniques and 15 tactics
- 2,267 multilabel examples (13.4%)
- 11,006 train / 2,550 validation / 3,391 test / 8 excluded near-duplicates
- zero exact-text, source or audited near-duplicate overlap across active partitions
- model progression: trivial baseline → Logistic Regression → Linear SVM → DistilBERT

Detailed split diagnostics, calibration tables, per-label results, feature weights and error exports are committed under `artifacts/metrics/`. See [experiment notes](docs/experiment_notes.md) for decisions and observed errors.

## Reproduce the experiment

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements-dev.txt
python -m src.data.build_dataset
python -m src.data.split
python -m src.models.train_classical
python -m src.models.calibrate
python -m src.evaluation.evaluate
python -m src.evaluation.error_analysis
python -m src.evaluation.technique_holdout
python -m src.models.package_inference
python -m pytest
```

To rerun DistilBERT, use `python -m src.models.train_transformer` or open [`notebooks/02_train_transformer_colab.ipynb`](notebooks/02_train_transformer_colab.ipynb) in Colab. The script uses the same persisted partitions, class-weighted loss, early stopping and validation-only threshold selection.
