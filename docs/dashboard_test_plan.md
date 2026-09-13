# Dashboard test plan

Run the dashboard from the repository root:

```bash
python -m pip install -r requirements-app.txt
streamlit run app.py
```

## Analyze report

Use the included examples so the checks are repeatable.

| Check | Example and action | Expected result |
|---|---|---|
| Clear single label | Load **PowerShell execution**, then analyze | Execution; accepted automatically; provenance and supporting words appear |
| Multilabel output | Load **Multi-tactic scheduled task**, then analyze | Execution, Privilege Escalation and Persistence |
| Abstention | Load **Ambiguous tool transfer**, then analyze | Needs analyst review; no suggestion is included; top candidate is shown |
| Routing control | Load **Routing threshold check** | Accepted at 70%; needs review at 90% without re-entering the text |
| Review actions | Analyze any example, then use both decision buttons | The report can be marked accepted or kept for review within the session |
| Empty input | Clear the text box and analyze | Warning asking for a description; no traceback |
| JSON export | Analyze any example and download the prediction | JSON contains `predictions`, `top_candidate`, `routing_confidence`, `routing_threshold` and `decision` |

The exact probabilities may change after retraining. Label selection and routing behavior should only change when updated evidence and artifacts are committed together.

## Review batch

1. Open **Review batch** and download the sample.
2. Upload the same file and select `text` as the description column.
3. Create the review queue. Four rows should be returned and at least the WellMess row should need review at the default threshold.
4. Change **Filter** to **Needs review** and confirm the table is filtered.
5. Select a table row and confirm its full report and result appear underneath.
6. Download the queue and check that the original columns plus `predicted_tactics`, `routing_confidence` and `decision` are present.

Also try these failure cases:

- CSV with an empty value in the selected text column;
- CSV with headers but no rows;
- CSV where the wrong column is selected;
- file larger than 2 MB;
- file containing more than 200 rows, which should be truncated with a notice.

## Model notes

Confirm that:

- the training summary shows 16,955 examples and 15 tactics;
- the model table lists the trivial baseline and three classical models;
- the single model-comparison figure renders;
- the limitations section reports 0.516 macro F1 for unseen techniques;
- the prior DistilBERT section is clearly marked as awaiting a clean-split rerun.

## Automated checks

```bash
python -m pytest
```

The test suite covers model loading, single-report interaction, batch transformation, CSV export safety, routing changes and inference schema. File-upload rendering and downloaded-file inspection remain manual checks.
