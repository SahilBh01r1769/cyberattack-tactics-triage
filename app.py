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
                "Included in result": bool(result["predictions"]),
            }
        )
    return pd.DataFrame(rows)


def render_prediction(result: dict, predictor: TacticPredictor, text: str, catalog: list[dict]) -> None:
    if result["decision"] == "auto_route":
        st.success(
            f"Accepted automatically · {result['routing_confidence']:.1%} confidence "
            f"(minimum: {result['routing_threshold']:.0%})"
        )
    else:
        st.warning(
            f"Needs analyst review · {result['routing_confidence']:.1%} confidence "
            f"(minimum: {result['routing_threshold']:.0%})"
        )

    st.markdown("#### Suggested ATT&CK tactics")
    if not result["predictions"]:
        st.caption("No suggestion was confident enough to include. The strongest candidate is shown for context.")
    st.dataframe(
        prediction_table(result, catalog),
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
            "Included in result": st.column_config.CheckboxColumn(),
        },
    )

    st.markdown("#### Words that influenced this result")
    st.caption("These words pushed the model toward each suggestion. They help inspect the result, but do not prove why an attack happened.")
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

    with st.expander("What these ATT&CK tactics mean"):
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
    st.subheader("Analyze one report")
    st.write("Paste a short threat description, or start with one of the real ATT&CK examples included below.")
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

    if st.button("Analyze report", type="primary"):
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
    st.subheader("Review several reports")
    st.write("Upload a CSV, choose the column that contains the descriptions, and receive a review queue. Up to 200 rows are processed at once.")
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
    if st.button("Analyze CSV", type="primary", disabled=predictor is None):
        try:
            st.session_state.batch_result = classify_frame(frame, text_column, predictor, threshold)
        except ValueError as exc:
            st.error(str(exc))

    result = st.session_state.get("batch_result")
    if result is not None:
        review_count = int(result["decision"].eq("analyst_review").sum())
        left, middle, right = st.columns(3)
        left.metric("Rows", len(result))
        middle.metric("Accepted", len(result) - review_count)
        right.metric("Needs review", review_count)
        review_only = st.checkbox("Show only reports that need review")
        display = result[result["decision"].eq("analyst_review")] if review_only else result
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "routing_confidence": st.column_config.ProgressColumn(
                    "Confidence", format="percent", min_value=0.0, max_value=1.0
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
    st.subheader("How the model performed")
    st.write("These results come from reports the models did not see during training. Sources were kept separate to make the comparison harder and more realistic.")
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
        "Macro F1 gives every tactic equal importance, including rare ones. Linear SVM scored highest overall. "
        "The Logistic Regression model is used in the workbench because its confidence scores were measured and calibrated."
    )

    chart_left, chart_right = st.columns(2)
    chart_left.image("artifacts/figures/model_comparison.png", caption="Classical model comparison")
    chart_right.image("artifacts/figures/confidence_coverage.png", caption="Routing threshold trade-off")

    stress = json.loads(Path("artifacts/metrics/technique_holdout_metrics.json").read_text())
    with st.expander("Harder test: techniques never seen during training"):
        st.write(
            f"The selected SVM was retrained on {stress['train_techniques']} techniques and evaluated on "
            f"{stress['holdout_techniques']} unseen techniques. Technique overlap: {stress['technique_overlap']}."
        )
        st.metric("Macro F1 on this harder test", f"{stress['metrics']['macro_f1']:.3f}")
        st.caption("This answers a harder question than the main test, so the two scores should not be compared directly.")

    with st.expander("Earlier transformer experiment"):
        st.warning("This DistilBERT run used an older data split, so it is shown as background evidence and not ranked with the current models.")
        transformer_left, transformer_right = st.columns(2)
        transformer_left.image("artifacts/figures/transformer_training_history.png", caption="Training history")
        transformer_right.image("artifacts/figures/transformer_per_label_comparison.png", caption="Per-label results")


st.set_page_config(page_title="Threat report triage", page_icon="◼", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .stApp { background: #26231f; color: #eee5d8; }
    .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1 { color: #f2e8da; font-size: 2.25rem; letter-spacing: -0.035em; }
    h2, h3, h4 { color: #e7dac7; }
    [data-testid="stHeader"] { background: rgba(38, 35, 31, 0.94); }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #312d28;
        border-color: #4d463c;
        box-shadow: none;
    }
    [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: #1f1d1a;
        border-radius: 0.4rem;
        padding: 0.32rem;
    }
    [data-baseweb="tab"] {
        color: #d8cbb8;
        border-radius: 0.25rem;
        padding-left: 1.15rem;
        padding-right: 1.15rem;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        color: #241f1a;
        background: #b99b76;
    }
    [data-baseweb="tab-highlight"] { display: none; }
    [data-testid="stMetric"] { background: transparent; }
    .stButton > button[kind="primary"] { color: #211d19; background: #b9976d; border-color: #b9976d; }
    .stButton > button[kind="primary"]:hover { color: #181512; background: #c9aa83; border-color: #c9aa83; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Threat report triage")
st.write("Turn a short cyber-threat report into suggested MITRE ATT&CK tactics, then decide whether the result is confident enough to use or needs a person to check it.")

catalog = load_json(CATALOG_PATH)
model_path = available_model()
predictor = load_predictor(str(model_path)) if model_path else None

with st.container(border=True):
    control, current, guidance = st.columns([2.2, 0.8, 1.5], vertical_alignment="center")
    with control:
        threshold = st.slider(
            "Confidence needed to accept a result",
            min_value=0.40,
            max_value=0.90,
            value=0.70,
            step=0.05,
            help="Raise this when you would rather review more reports than accept uncertain suggestions.",
        )
    current.metric("Current minimum", f"{threshold:.0%}")
    guidance.caption("Higher settings accept fewer reports automatically and send more to review. At 70%, the test accepted 78% of reports.")

if predictor is None:
    st.error("The saved model could not be loaded. Run `python -m src.models.package_inference` to rebuild it.")

single_tab, batch_tab, results_tab = st.tabs(["Analyze a report", "Review a CSV", "Model evidence"])
with single_tab:
    single_report(predictor, threshold, catalog)
with batch_tab:
    batch_queue(predictor, threshold)
with results_tab:
    experiment_results()

st.divider()
st.caption("Decision-support prototype. Predictions should be reviewed alongside the original intelligence context.")
