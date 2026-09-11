# Experiment notes

## Target choice

The task remains multilabel tactic classification. The processed snapshot has 16,955 examples, 15 valid tactic labels and 2,267 examples (13.4%) attached to techniques that span multiple tactics. All labels appear in each split; the smallest, `reconnaissance`, still has 112/34/29 positive examples in train/validation/test. Converting this to one primary tactic would discard real labels without solving a data-volume problem.

Technique classification was not attempted: 611 technique classes would make the tail too sparse for this first version.

## Text and provenance

Procedure text comes only from active Enterprise ATT&CK `uses` relationships. HTML, Markdown link destinations and inline citation markers are removed; linked entity names remain. Citations, relationship IDs, technique IDs and source entities are retained in separate columns. After normalization, 104 exact duplicates were removed. A nearest-neighbor word n-gram check found 39 pairs at cosine similarity >= 0.92, affecting 61 examples. This is an estimate rather than a full all-pairs deduplication pass.

## Split decision

The final split groups by ATT&CK source entity (`source_id`), so an intrusion set, malware family, tool or campaign cannot occur in more than one partition. Sixty-four deterministic group-shuffle candidates were evaluated for each holdout, choosing the candidate with the smallest size and label-prevalence drift while requiring every tactic in the holdout.

The active result is 11,006 train, 2,550 validation and 3,391 test examples. Eight records are excluded because they formed cross-partition pairs above 0.92 word n-gram cosine similarity. The exclusion policy preserves test over validation and validation over training, preventing cleanup from making evaluation easier. The post-cleanup audit has zero exact-text, source and flagged near-duplicate overlap. A simple random 80/20 comparison produced 882 overlapping sources.

Techniques are allowed to cross the primary partitions: the intended main test is generalization to unseen actors/software using known behavior categories. A separate technique-held-out stress test uses 8,748 training examples from 452 techniques and 2,258 examples from 114 unseen techniques. SVM macro F1 falls from 0.773 on the primary test to 0.516 under this harder condition, with zero technique overlap. Source overlap is allowed in that diagnostic so only the technique-generalization axis is changed.

## Classical models

Ten limited configurations compared word unigram/bigram and combined word/character TF-IDF, class weighting and a few regularization strengths. The selected models were chosen on validation macro F1 and evaluated once on the grouped test set.

- Best Logistic Regression: word + character TF-IDF, `C=2`, balanced class weights. Test macro F1 0.742.
- Best Linear SVM: word + character TF-IDF, `C=1`, balanced class weights. Test macro F1 0.773.
- The unweighted LR variant was materially worse on validation macro F1 (0.594), supporting class weighting for the long-tailed labels.

The SVM is the accuracy-oriented model. It is not presented as probabilistic. Logistic Regression is used for calibrated confidence and inference.

## Calibration and routing

The validation partition is divided into source-disjoint calibration (1,274 examples) and threshold-selection (1,276 examples) subsets. Independent Platt scalers are fitted on the first subset; per-label F1 thresholds are selected only on the second. On test data, calibration reduced macro Brier score from 0.0266 to 0.0213 and macro expected calibration error from 0.0416 to 0.0085. The default analyst-routing threshold of 0.70 accepts 78.0% of examples; accepted examples reach 0.878 micro F1 and 0.897 samples F1.

Routing confidence is the highest calibrated probability among predicted tactics. This answers whether at least one routing decision is strong; it does not prove the predicted label set is complete.

## Error patterns observed

- The SVM made at least one label error on 949 of 3,391 test records, and emitted no positive label on 238. The largest missed-with-no-replacement counts were `stealth` (62), `execution` (37) and `discovery` (36).
- `collection` and `discovery` were substituted for each other 13 times in each direction. Short descriptions about locating and then acquiring data often contain vocabulary supporting both.
- Context words can conflict with the technique-derived label. One `Malicious File` example describes spearphishing delivery, so the model predicted `initial-access` while its linked technique contributes `execution`.
- Multitactic techniques are sometimes only partially recovered. A `Modify Registry` example linked to `defense-impairment|persistence` was assigned only `defense-impairment`; a `Time Based Checks` example linked to `discovery|stealth` was assigned only `stealth`.
- Descriptions can genuinely suggest additional behavior. A `Cloud Accounts` record received `credential-access` in addition to its four reference labels because it explicitly says the actor gained access to an administrator account.
- Sparse `reconnaissance` remains the weakest SVM label (test F1 0.553). Its test estimate is based on only 29 positives and should be treated cautiously.

## Transformer status

The recorded transformer run predates the eight-record split cleanup. Its figures and metrics remain committed as historical evidence, but the current comparison script rejects it because its recorded split sizes no longer match the active partitions. It must be rerun before being reported alongside the cleaned classical results.

The completed DistilBERT run used multi-hot labels, positive-class weights, best-checkpoint loading and the shared grouped partitions. It trained for three epochs on a Tesla T4 in 106 seconds. Validation loss decreased at every epoch (0.482, 0.367, 0.349), while fixed-threshold validation macro F1 increased (0.568, 0.650, 0.705), so early stopping did not activate.

Positive weighting made the default 0.5 threshold recall-heavy: test recall was 0.902, but precision was 0.656 and macro F1 was 0.695. Per-label thresholds selected on validation data produced 0.824 precision, 0.819 recall and 0.779 macro F1 on test. The large threshold effect is part of the result and is not hidden.

DistilBERT narrowly exceeded Linear SVM in macro F1 (0.779 vs 0.772), samples F1 (0.819 vs 0.807) and exact label-set accuracy (0.753 vs 0.720). SVM remained slightly stronger in micro F1 (0.826 vs 0.822) and Hamming loss (0.02737 vs 0.02770). The transformer improved most on `resource-development` (+0.094 F1), `impact` (+0.079) and `exfiltration` (+0.060), but lost most on `persistence` (-0.058).

The transformer made at least one label error on 836 of 3,391 test examples and emitted no tactic on 122. It performed better on single-label records than multilabel records (samples F1 0.826 vs 0.775), showing that complete recovery of tactics attached to multi-tactic techniques remains a weakness. Frequent residual confusions include `stealth` versus `defense-impairment` or `persistence`.
