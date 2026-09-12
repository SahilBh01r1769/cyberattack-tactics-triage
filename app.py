from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard import classify_frame, format_tactic, load_json_records, safe_csv_bytes
from src.inference.predict import TacticPredictor


MODEL_CANDIDATES = [
    Path("artifacts/models/triage_lr.joblib"),
    Path("artifacts/models/confidence_logistic_regression.joblib"),
]
CATALOG_PATH = "data/tactic_catalog.json"
EXAMPLES_PATH = "data/demo_examples.json"
SAMPLE_BATCH_PATH = Path("data/demo_batch.csv")


@st.cache_resource
def load_predictor(path: str) -> TacticPredictor:
    return TacticPredictor.from_artifact(path)


@st.cache_data
def load_json(path: str) -> list[dict]:
    return load_json_records(path)


def available_model() -> Path | None:
    return next((path for path in MODEL_CANDIDATES if path.exists()), None)


def prediction_table(result: dict, catalog: list[dict]) -> pd.DataFrame:
    catalog_by_slug = {item["slug"]: item for item in catalog}
    predictions = result["predictions"] or [result["top_candidate"]]
    rows = []
    for prediction in predictions:
        metadata = catalog_by_slug[prediction["tactic"]]
        rows.append(
            {
                "Tactic": metadata["name"],
                "ATT&CK ID": metadata["external_id"],
                "Confidence": prediction["confidence"],
                "Passed label threshold": bool(result["predictions"]),
            }
        )
    return pd.DataFrame(rows)


def render_prediction(result: dict, predictor: TacticPredictor, text: str, catalog: list[dict]) -> None:
    if result["decision"] == "auto_route":
        st.success(
            f"Auto-route · {result['routing_confidence']:.1%} routing confidence "
            f"(required: {result['routing_threshold']:.0%})"
        )
    else:
        st.warning(
            f"Analyst review recommended · {result['routing_confidence']:.1%} routing confidence "
            f"(required: {result['routing_threshold']:.0%})"
        )

    st.markdown("#### Predicted tactics")
    if not result["predictions"]:
        st.caption("No tactic passed its learned label threshold. The strongest candidate is shown for context.")
    st.dataframe(
        prediction_table(result, catalog),
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
            "Passed label threshold": st.column_config.CheckboxColumn(),
        },
    )

    st.markdown("#### Terms supporting the prediction")
    st.caption("Positive contributions from the deployed TF-IDF Logistic Regression model; they are not causal explanations.")
    predictions = result["predictions"] or [result["top_candidate"]]
    for index, prediction in enumerate(predictions[:3]):
        tactic = prediction["tactic"]
        contributions = predictor.explain(text, tactic)
        with st.expander(format_tactic(tactic), expanded=index == 0):
            if contributions:
                evidence = pd.DataFrame(contributions).rename(
                    columns={"feature": "Term", "contribution": "Positive contribution"}
                )
                st.dataframe(
                    evidence,
                    width="stretch",
                    hide_index=True,
                    column_config={"Positive contribution": st.column_config.NumberColumn(format="%.3f")},
                )
            else:
                st.caption("No positive feature contribution was available for this input.")

    with st.expander("ATT&CK tactic reference"):
        reference = pd.DataFrame(catalog)[["external_id", "name", "description", "url"]]
        reference.columns = ["ID", "Tactic", "Description", "Reference"]
        st.dataframe(
            reference,
            width="stretch",
            hide_index=True,
            column_config={"Reference": st.column_config.LinkColumn(display_text="MITRE")},
        )

    st.download_button(
        "Download prediction (JSON)",
        json.dumps(result, indent=2),
        file_name="attack_triage_result.json",
        mime="application/json",
    )


def single_report(predictor: TacticPredictor | None, threshold: float, catalog: list[dict]) -> None:
    st.subheader("Single report")
    examples = load_json(EXAMPLES_PATH)
    selected = st.selectbox(
        "Load an ATT&CK-derived example",
        range(len(examples)),
        format_func=lambda index: examples[index]["name"],
    )
    if st.button("Use selected example"):
        st.session_state.threat_text = examples[selected]["text"]
        st.session_state.example_source = examples[selected]

    text = st.text_area(
        "Threat or incident description",
        key="threat_text",
        height=150,
        placeholder="Example: The adversary used PowerShell to execute a downloaded payload.",
    )
    source = st.session_state.get("example_source")
    if source and source["text"] == text:
        st.caption(
            f"ATT&CK example provenance: {source['technique_id']} {source['technique_name']} · "
            f"source entity: {source['source_name']} ({source['source_id']})"
        )

    if st.button("Classify report", type="primary"):
        if predictor is None:
            st.error("The model artifact is unavailable. Regenerate it using the command shown in the sidebar.")
        elif not text.strip():
            st.warning("Enter a threat description first.")
        else:
            st.session_state.last_result = predictor.predict(text, threshold)
            st.session_state.last_result_text = text

    result = st.session_state.get("last_result")
    if result and st.session_state.get("last_result_text") == text and predictor is not None:
        if result["routing_threshold"] != threshold:
            result = predictor.predict(text, threshold)
            st.session_state.last_result = result
        st.divider()
        render_prediction(result, predictor, text, catalog)


def batch_queue(predictor: TacticPredictor | None, threshold: float) -> None:
    st.subheader("Batch CSV")
    st.write("Upload a CSV and choose the column containing threat descriptions. The demo processes at most 200 rows.")
    st.download_button(
        "Download sample CSV",
        SAMPLE_BATCH_PATH.read_bytes(),
        file_name="attack_triage_sample.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return
    if upload.size > 2_000_000:
        st.error("The file must be smaller than 2 MB.")
        return
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    if frame.empty:
        st.warning("The uploaded CSV has no rows.")
        return
    if len(frame) > 200:
        st.info("Only the first 200 rows will be classified.")
        frame = frame.head(200)

    text_column = st.selectbox("Description column", frame.columns.tolist())
    st.dataframe(frame.head(5), width="stretch", hide_index=True)
    if st.button("Classify CSV", type="primary", disabled=predictor is None):
        try:
            st.session_state.batch_result = classify_frame(frame, text_column, predictor, threshold)
        except ValueError as exc:
            st.error(str(exc))

    result = st.session_state.get("batch_result")
    if result is not None:
        review_count = int(result["decision"].eq("analyst_review").sum())
        left, middle, right = st.columns(3)
        left.metric("Rows", len(result))
        middle.metric("Auto-routed", len(result) - review_count)
        right.metric("Review queue", review_count)
        review_only = st.checkbox("Show analyst-review rows only")
        display = result[result["decision"].eq("analyst_review")] if review_only else result
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "routing_confidence": st.column_config.ProgressColumn(
                    "Routing confidence", format="percent", min_value=0.0, max_value=1.0
                )
            },
        )
        st.download_button(
            "Download classified CSV",
            safe_csv_bytes(result),
            file_name="attack_triage_queue.csv",
            mime="text/csv",
        )


def experiment_results() -> None:
    st.subheader("Experiment results")
    st.caption("All figures and tables below are generated from committed held-out predictions.")
    stats = json.loads(Path("artifacts/metrics/dataset_statistics.json").read_text())
    left, middle_left, middle_right, right = st.columns(4)
    left.metric("Procedure examples", f"{stats['usable_samples']:,}")
    middle_left.metric("Techniques", stats["techniques"])
    middle_right.metric("Tactics", stats["tactics"])
    right.metric("Multilabel records", f"{stats['multilabel_fraction']:.1%}")

    comparison = pd.read_csv("artifacts/metrics/model_comparison.csv")
    comparison = comparison[["model", "macro_f1", "micro_f1", "samples_f1", "subset_accuracy"]]
    comparison.columns = ["Model", "Macro F1", "Micro F1", "Samples F1", "Exact match"]
    st.markdown("#### Grouped-source test set")
    st.dataframe(
        comparison.style.format({column: "{:.3f}" for column in comparison.columns[1:]}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Linear SVM is the strongest verified classifier. Calibrated Logistic Regression is deployed because "
        "it provides evaluated confidence estimates and interpretable text features."
    )

    chart_left, chart_right = st.columns(2)
    chart_left.image("artifacts/figures/model_comparison.png", caption="Classical model comparison")
    chart_right.image("artifacts/figures/confidence_coverage.png", caption="Routing threshold trade-off")

    stress = json.loads(Path("artifacts/metrics/technique_holdout_metrics.json").read_text())
    with st.expander("Unseen-technique stress test"):
        st.write(
            f"The selected SVM was retrained on {stress['train_techniques']} techniques and evaluated on "
            f"{stress['holdout_techniques']} unseen techniques. Technique overlap: {stress['technique_overlap']}."
        )
        st.metric("Stress-test macro F1", f"{stress['metrics']['macro_f1']:.3f}")
        st.caption("This diagnostic changes the generalization target and is not directly comparable to the main test.")

    with st.expander("Prior DistilBERT run"):
        st.warning("This run predates the eight-record split cleanup and is excluded from the current comparison.")
        transformer_left, transformer_right = st.columns(2)
        transformer_left.image("artifacts/figures/transformer_training_history.png", caption="Training history")
        transformer_right.image("artifacts/figures/transformer_per_label_comparison.png", caption="Per-label results")


st.set_page_config(page_title="ATT&CK tactic classifier", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #ded8ce; }
    h1 { font-size: 2.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("ATT&CK tactic classifier")
st.caption("A small NLP workbench for mapping short cyber-threat descriptions to Enterprise ATT&CK tactics.")

catalog = load_json(CATALOG_PATH)
model_path = available_model()
predictor = load_predictor(str(model_path)) if model_path else None

with st.sidebar:
    st.header("Model controls")
    threshold = st.slider(
        "Auto-route threshold",
        min_value=0.40,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="Higher values send more cases to analyst review.",
    )
    st.caption("At 0.70: 78.0% test coverage and 0.878 micro F1 on accepted records.")
    st.divider()
    st.write("**Inference model**")
    st.caption("TF-IDF + calibrated Logistic Regression")
    st.write("**Best verified classifier**")
    st.caption("TF-IDF + Linear SVM · 0.773 macro F1")
    if predictor is None:
        st.error("Model artifact missing")
        st.code("python -m src.models.package_inference", language="bash")

single_tab, batch_tab, results_tab = st.tabs(["Single report", "Batch CSV", "Experiment results"])
with single_tab:
    single_report(predictor, threshold, catalog)
with batch_tab:
    batch_queue(predictor, threshold)
with results_tab:
    experiment_results()

st.divider()
st.caption("Decision-support prototype. Predictions should be reviewed alongside the original intelligence context.")
