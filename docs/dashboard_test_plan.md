# Dashboard test plan

Run the dashboard from the repository root:

```bash
python -m pip install -r requirements-app.txt
streamlit run app.py
```

## Single report

Use the included examples so the checks are repeatable.

| Check | Example and action | Expected result |
|---|---|---|
| Clear single label | Load **PowerShell execution**, then classify | Execution; auto-route; provenance and supporting terms appear |
| Multilabel output | Load **Multi-tactic scheduled task**, then classify | Execution, Privilege Escalation and Persistence |
| Abstention | Load **Ambiguous tool transfer**, then classify | Analyst review; no label passes its learned threshold; top candidate is shown |
| Routing control | Load **Routing threshold check** | Auto-route at 0.70; analyst review at 0.90 without re-entering the text |
| Empty input | Clear the text box and classify | Warning asking for a description; no traceback |
| JSON export | Classify any example and download the prediction | JSON contains `predictions`, `top_candidate`, `routing_confidence`, `routing_threshold` and `decision` |

The exact probabilities may change after retraining. Label selection and routing behavior should only change when updated evidence and artifacts are committed together.

## Batch CSV

1. Open **Batch CSV** and download the sample.
2. Upload the same file and select `text` as the description column.
3. Classify it. Four rows should be returned and at least the WellMess row should enter the review queue at the default threshold.
4. Enable **Show analyst-review rows only** and confirm the table is filtered.
5. Download the classified CSV and check that the original columns plus `predicted_tactics`, `routing_confidence` and `decision` are present.

Also try these failure cases:

- CSV with an empty value in the selected text column;
- CSV with headers but no rows;
- CSV where the wrong column is selected;
- file larger than 2 MB;
- file containing more than 200 rows, which should be truncated with a notice.

## Experiment results

Confirm that:

- the dataset summary shows 16,955 examples, 611 techniques and 15 tactics;
- the model table lists the trivial baseline and three classical models;
- both committed figures render;
- the unseen-technique section reports 114 held-out techniques, zero overlap and 0.516 macro F1;
- the prior DistilBERT section is clearly marked as awaiting a clean-split rerun.

## Automated checks

```bash
python -m pytest
```

The test suite covers model loading, single-report interaction, batch transformation, CSV export safety, routing changes and inference schema. File-upload rendering and downloaded-file inspection remain manual checks.
