# Experiment notes

## Target choice

The task remains multilabel tactic classification. The processed snapshot has 16,955 examples, 15 valid tactic labels and 2,267 examples (13.4%) attached to techniques that span multiple tactics. All labels appear in each split; the smallest, `reconnaissance`, still has 112/34/29 positive examples in train/validation/test. Converting this to one primary tactic would discard real labels without solving a data-volume problem.

Technique classification was not attempted: 611 technique classes would make the tail too sparse for this first version.

## Text and provenance

Procedure text comes only from active Enterprise ATT&CK `uses` relationships. HTML, Markdown link destinations and inline citation markers are removed; linked entity names remain. Citations, relationship IDs, technique IDs and source entities are retained in separate columns. After normalization, 104 exact duplicates were removed. A nearest-neighbor word n-gram check found 39 pairs at cosine similarity >= 0.92, affecting 61 examples. This is an estimate rather than a full all-pairs deduplication pass.

## Split decision

The final split groups by ATT&CK source entity (`source_id`), so an intrusion set, malware family, tool or campaign cannot occur in more than one partition. Sixty-four deterministic group-shuffle candidates were evaluated for each holdout, choosing the candidate with the smallest size and label-prevalence drift while requiring every tactic in the holdout.

The result is 11,013 train, 2,551 validation and 3,391 test examples, with zero exact-text and zero source overlap. A simple random 80/20 comparison produced 882 overlapping sources. An approximate nearest-neighbor audit found 8 cross-partition pairs above 0.92 word n-gram cosine similarity; their relationship IDs are saved in `split_diagnostics.json`. They were documented rather than used to repeatedly tune the split. Techniques are allowed to cross partitions: the intended test is generalization to unseen actors/software using known behavior categories, not zero-shot generalization to unseen techniques.

## Classical models

Ten limited configurations compared word unigram/bigram and combined word/character TF-IDF, class weighting and a few regularization strengths. The selected models were chosen on validation macro F1 and evaluated once on the grouped test set.

- Best Logistic Regression: word + character TF-IDF, `C=2`, balanced class weights. Test macro F1 0.741.
- Best Linear SVM: word + character TF-IDF, `C=1`, balanced class weights. Test macro F1 0.772.
- The unweighted LR variant was materially worse on validation macro F1 (0.594), supporting class weighting for the long-tailed labels.

The SVM is the accuracy-oriented model. It is not presented as probabilistic. Logistic Regression is used for calibrated confidence and inference.

## Calibration and routing

Independent Platt scalers and per-label F1 thresholds are fitted on the validation partition. On test data, calibration reduced macro Brier score from 0.0265 to 0.0213 and macro expected calibration error from 0.0416 to 0.0077. The default analyst-routing threshold of 0.70 accepts 77.6% of examples; accepted examples reach 0.884 micro F1 and 0.901 samples F1.

Routing confidence is the highest calibrated probability among predicted tactics. This answers whether at least one routing decision is strong; it does not prove the predicted label set is complete.

## Error patterns observed

- The SVM made at least one label error on 949 of 3,391 test records, and emitted no positive label on 238. The largest missed-with-no-replacement counts were `stealth` (62), `execution` (37) and `discovery` (36).
- `collection` and `discovery` were substituted for each other 13 times in each direction. Short descriptions about locating and then acquiring data often contain vocabulary supporting both.
- Context words can conflict with the technique-derived label. One `Malicious File` example describes spearphishing delivery, so the model predicted `initial-access` while its linked technique contributes `execution`.
- Multitactic techniques are sometimes only partially recovered. A `Modify Registry` example linked to `defense-impairment|persistence` was assigned only `defense-impairment`; a `Time Based Checks` example linked to `discovery|stealth` was assigned only `stealth`.
- Descriptions can genuinely suggest additional behavior. A `Cloud Accounts` record received `credential-access` in addition to its four reference labels because it explicitly says the actor gained access to an administrator account.
- Sparse `reconnaissance` remains the weakest SVM label (test F1 0.50). Its test estimate is based on only 29 positives and should be treated cautiously.

## Transformer status

The DistilBERT path uses multi-hot labels, positive-class weights, early stopping, best-checkpoint loading and the shared grouped partitions. A 128-example smoke run was attempted, but the model download timed out and the machine had no GPU. The attempt was stopped rather than spending the run on infrastructure retries. Full training and comparison remain open.
