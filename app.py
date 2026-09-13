from __future__ import annotations

import html
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


def result_rows(result: dict, catalog: list[dict]) -> list[dict]:
    catalog_by_slug = {item["slug"]: item for item in catalog}
    predictions = result["predictions"] or [result["top_candidate"]]
    return [
        {
            "Tactic": catalog_by_slug[item["tactic"]]["name"],
            "ATT&CK ID": catalog_by_slug[item["tactic"]]["external_id"],
            "Confidence": item["confidence"],
        }
        for item in predictions
    ]


def render_decision_sheet(result: dict | None, catalog: list[dict]) -> None:
    st.markdown('<p class="section-label">Routing decision</p>', unsafe_allow_html=True)
    if result is None:
        st.markdown(
            '<div class="empty-decision">The routing decision and suggested tactics will appear here.</div>',
            unsafe_allow_html=True,
        )
        return

    threshold = result["routing_threshold"]
    confidence = result["routing_confidence"]
    if result["decision"] == "auto_route":
        st.markdown('<p class="decision accepted">Accepted automatically</p>', unsafe_allow_html=True)
        st.caption(f"Confidence reached the {threshold:.0%} requirement.")
    else:
        st.markdown('<p class="decision review">Needs analyst review</p>', unsafe_allow_html=True)
        st.caption(f"Confidence did not reach the {threshold:.0%} requirement.")

    st.markdown("#### Suggested tactics")
    if not result["predictions"]:
        st.caption("No tactic was confident enough to include. The strongest candidate is shown below.")
    st.dataframe(
        pd.DataFrame(result_rows(result, catalog)),
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
        },
    )
    st.caption(f"Routing confidence: {confidence:.1%}")

    accept, review = st.columns(2)
    if accept.button("Accept suggestion", width="stretch"):
        st.session_state.review_action = "accepted"
    if review.button("Keep for review", width="stretch"):
        st.session_state.review_action = "review"
    if st.session_state.get("review_action") == "accepted":
        st.caption("Marked as accepted for this session.")
    elif st.session_state.get("review_action") == "review":
        st.caption("Kept in the review queue for this session.")


def render_influential_terms(result: dict, predictor: TacticPredictor, text: str) -> None:
    predictions = result["predictions"] or [result["top_candidate"]]
    for prediction in predictions[:3]:
        tactic = prediction["tactic"]
        contributions = predictor.explain(text, tactic)[:5]
        st.markdown(f"**{format_tactic(tactic)}**")
        if contributions:
            evidence = pd.DataFrame(contributions).rename(
                columns={"feature": "Term", "contribution": "Contribution"}
            )
            st.dataframe(
                evidence,
                width="stretch",
                hide_index=True,
                column_config={"Contribution": st.column_config.NumberColumn(format="+%.3f")},
            )
        else:
            st.caption("No positive term contribution was available for this suggestion.")


def render_lifecycle(result: dict, catalog: list[dict]) -> None:
    highlighted = {item["tactic"] for item in result["predictions"]}
    if not highlighted:
        highlighted = {result["top_candidate"]["tactic"]}
    items = []
    for tactic in catalog:
        selected = " selected" if tactic["slug"] in highlighted else ""
        items.append(
            f'<span class="lifecycle-item{selected}" title="{html.escape(tactic["description"])}">'
            f'{html.escape(tactic["name"])}</span>'
        )
    st.markdown('<div class="lifecycle-strip">' + "".join(items) + "</div>", unsafe_allow_html=True)
    st.caption("Highlighted stages are the model suggestions. Hover over a stage for its ATT&CK description.")


def render_explanations(
    result: dict,
    predictor: TacticPredictor,
    text: str,
    catalog: list[dict],
    source: dict | None,
) -> None:
    st.markdown("### Review the evidence")
    with st.expander("Why these tactics were suggested"):
        st.caption("The strongest words and phrases from the deployed text model. These are clues, not proof.")
        render_influential_terms(result, predictor, text)

    with st.expander("Position in the ATT&CK lifecycle"):
        render_lifecycle(result, catalog)

    with st.expander("Original example and provenance"):
        st.write(text)
        if source and source["text"] == text:
            st.caption(
                f"MITRE ATT&CK procedure example · {source['technique_id']} {source['technique_name']} · "
                f"source: {source['source_name']} ({source['source_id']})"
            )
        else:
            st.caption("User-provided report. No ATT&CK source metadata is attached.")
        st.download_button(
            "Download result as JSON",
            json.dumps(result, indent=2),
            file_name="attack_triage_result.json",
            mime="application/json",
        )


def analyze_report(predictor: TacticPredictor | None, catalog: list[dict]) -> None:
    st.subheader("Analyze report")
    report_column, decision_column = st.columns([1.4, 1], gap="large")

    with report_column:
        st.markdown('<p class="section-label">Threat report</p>', unsafe_allow_html=True)
        examples = load_json(EXAMPLES_PATH)
        selected = st.selectbox(
            "Select an ATT&CK example",
            range(len(examples)),
            format_func=lambda index: examples[index]["name"],
        )
        if st.button("Load example"):
            st.session_state.threat_text = examples[selected]["text"]
            st.session_state.example_source = examples[selected]
            st.session_state.review_action = None

        text = st.text_area(
            "Report text",
            key="threat_text",
            height=180,
            placeholder="Paste or edit a short threat description here.",
            label_visibility="collapsed",
        )
        source = st.session_state.get("example_source")
        if source and source["text"] == text:
            st.caption(
                f"Source: ATT&CK procedure example · {source['technique_id']} {source['technique_name']} · "
                f"{source['source_name']}"
            )
        else:
            st.caption("Source: user-provided report")

        action_space, action = st.columns([1.7, 1])
        with action:
            analyze = st.button("Analyze report", type="primary", width="stretch")

        st.markdown('<p class="policy-label">Review policy</p>', unsafe_allow_html=True)
        threshold_percent = st.slider(
            "Confidence needed for automatic routing",
            min_value=40,
            max_value=90,
            value=70,
            step=5,
            key="routing_threshold_percent",
            format="%d%%",
            help="Higher settings send more uncertain reports to a person for review.",
        )
        threshold = threshold_percent / 100

        if analyze:
            if predictor is None:
                st.error("The saved model is unavailable. Rebuild it before running the workbench.")
            elif not text.strip():
                st.warning("Enter a threat description first.")
            else:
                st.session_state.last_result = predictor.predict(text, threshold)
                st.session_state.last_result_text = text
                st.session_state.review_action = None

    result = st.session_state.get("last_result")
    if result and st.session_state.get("last_result_text") == text and predictor is not None:
        if result["routing_threshold"] != threshold:
            result = predictor.predict(text, threshold)
            st.session_state.last_result = result
            st.session_state.review_action = None
    else:
        result = None

    with decision_column:
        with st.container(border=True):
            render_decision_sheet(result, catalog)

    if result is not None and predictor is not None:
        st.divider()
        render_explanations(result, predictor, text, catalog, source)


def batch_queue(predictor: TacticPredictor | None) -> None:
    st.subheader("Batch review")
    st.write("Upload a CSV of short threat reports to create a simple review queue. Up to 200 rows are processed at once.")
    download, upload_column = st.columns([1, 2.2], vertical_alignment="bottom")
    with download:
        st.download_button(
            "Download example CSV",
            SAMPLE_BATCH_PATH.read_bytes(),
            file_name="attack_triage_sample.csv",
            mime="text/csv",
            width="stretch",
        )
    with upload_column:
        upload = st.file_uploader("Upload report file", type="csv")

    if upload is None:
        st.caption("The file should contain one column with the report text. Download the example to see the format.")
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
        st.info("Only the first 200 rows will be analyzed.")
        frame = frame.head(200)

    text_column = st.selectbox("Column containing the report", frame.columns.tolist())
    if st.button("Create review queue", type="primary", disabled=predictor is None):
        try:
            threshold = float(st.session_state.get("routing_threshold_percent", 70)) / 100
            st.session_state.batch_result = classify_frame(frame, text_column, predictor, threshold)
        except ValueError as exc:
            st.error(str(exc))

    result = st.session_state.get("batch_result")
    if result is None:
        st.dataframe(frame.head(5), width="stretch", hide_index=True)
        return

    review_count = int(result["decision"].eq("analyst_review").sum())
    st.markdown(
        f'<p class="queue-summary"><strong>{len(result)}</strong> reports &nbsp;&nbsp; '
        f'<strong>{len(result) - review_count}</strong> accepted &nbsp;&nbsp; '
        f'<strong>{review_count}</strong> need review</p>',
        unsafe_allow_html=True,
    )
    filter_choice = st.selectbox("Filter", ["All reports", "Needs review", "Accepted"])
    if filter_choice == "Needs review":
        display = result[result["decision"].eq("analyst_review")]
    elif filter_choice == "Accepted":
        display = result[result["decision"].eq("auto_route")]
    else:
        display = result

    display = display[[text_column, "predicted_tactics", "routing_confidence", "decision"]].copy()
    display["decision"] = display["decision"].map(
        {"auto_route": "Accepted", "analyst_review": "Needs review"}
    )
    display.columns = ["Report", "Suggested tactics", "Confidence", "Decision"]
    event = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={"Confidence": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)},
    )
    st.download_button(
        "Download review queue",
        safe_csv_bytes(result),
        file_name="attack_triage_queue.csv",
        mime="text/csv",
    )

    selected_rows = event.selection.rows
    if selected_rows:
        row = display.iloc[selected_rows[0]]
        with st.container(border=True):
            st.markdown("#### Selected report")
            st.write(row["Report"])
            st.caption(
                f"Suggested tactics: {row['Suggested tactics']} · Confidence: {row['Confidence']:.1%} · "
                f"Decision: {row['Decision']}"
            )


def model_notes() -> None:
    stats = json.loads(Path("artifacts/metrics/dataset_statistics.json").read_text())
    comparison = pd.read_csv("artifacts/metrics/model_comparison.csv")
    comparison = comparison[["model", "macro_f1", "micro_f1", "subset_accuracy"]]
    comparison.columns = ["Model", "Macro F1", "Micro F1", "Exact match"]
    stress = json.loads(Path("artifacts/metrics/technique_holdout_metrics.json").read_text())

    st.subheader("Model notes")
    st.markdown("### What was trained?")
    st.markdown(
        f"- **{stats['usable_samples']:,}** official ATT&CK procedure examples\n"
        f"- **{stats['tactics']}** possible tactics; one report may have several\n"
        "- Training and test data grouped by source, so the same actor, malware or campaign does not cross the split\n"
        "- No generated threat descriptions and no external classification API"
    )

    st.markdown("### How well did it perform?")
    st.dataframe(
        comparison.style.format({column: "{:.3f}" for column in comparison.columns[1:]}),
        width="stretch",
        hide_index=True,
    )
    st.image("artifacts/figures/model_comparison.png", caption="Performance on the source-grouped test set", width=720)
    st.info(
        "Linear SVM achieved the strongest classification score. Calibrated Logistic Regression powers the "
        "workbench because confidence-based routing needs measured probabilities."
    )

    st.markdown("### When should it not be trusted?")
    st.markdown(
        f"- On techniques never seen during training, macro F1 fell to **{stress['metrics']['macro_f1']:.3f}**.\n"
        "- Rare tactics have fewer examples, so their scores are less stable.\n"
        "- Confidence helps choose what to review; it does not guarantee that a suggestion is correct.\n"
        "- Short or vague descriptions can omit the context needed to distinguish related tactics.\n"
        "- A person should still check the original report before acting on a prediction."
    )
    with st.expander("Earlier transformer experiment"):
        st.write(
            "A previous DistilBERT run used an older split. It remains in the repository as historical evidence, "
            "but is not ranked against the cleaned results above until it is rerun."
        )


st.set_page_config(page_title="ATT&CK report triage", page_icon="◼", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .stApp { background: #24211e; color: #eee6da; }
    .block-container { max-width: 1120px; padding-top: 1.8rem; padding-bottom: 3rem; }
    h1 { color: #f1e8da; font-size: 2.15rem; letter-spacing: -0.035em; margin-bottom: 0.25rem; }
    h2, h3, h4 { color: #e8ddcc; letter-spacing: -0.015em; }
    [data-testid="stHeader"] { background: rgba(36, 33, 30, 0.96); }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #302c27;
        border: 1px solid #50483f;
        border-radius: 3px;
        box-shadow: none;
    }
    [data-baseweb="tab-list"] {
        gap: 0;
        margin-top: 1.1rem;
        background: #1d1b18;
        border-top: 1px solid #51483d;
        border-bottom: 1px solid #51483d;
        padding: 0 0.3rem;
    }
    [data-baseweb="tab"] {
        color: #cfc3b2;
        border-radius: 0;
        padding: 0.65rem 1.35rem;
    }
    [data-baseweb="tab"][aria-selected="true"] { color: #171411; background: #b39468; }
    [data-baseweb="tab-highlight"] { display: none; }
    .section-label {
        color: #b9aa95;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        margin: 0 0 0.65rem 0;
        text-transform: uppercase;
    }
    .policy-label { color: #c9baa5; font-size: 0.82rem; font-weight: 650; margin: 0.6rem 0 -0.5rem; }
    .empty-decision { color: #ad9f8e; min-height: 255px; padding-top: 5rem; text-align: center; }
    .decision { font-family: Georgia, serif; font-size: 1.65rem; line-height: 1.15; margin: 0.25rem 0 0; }
    .decision.accepted { color: #d0ad77; }
    .decision.review { color: #c27d72; }
    .queue-summary { color: #d9cebf; border-top: 1px solid #51483d; border-bottom: 1px solid #51483d; padding: 0.75rem 0; }
    .lifecycle-strip { display: flex; flex-wrap: wrap; gap: 0.32rem; }
    .lifecycle-item {
        color: #ad9f8e;
        border: 1px solid #51493f;
        border-radius: 2px;
        font-size: 0.76rem;
        padding: 0.38rem 0.52rem;
    }
    .lifecycle-item.selected { color: #1d1915; background: #b39468; border-color: #b39468; font-weight: 650; }
    .stButton > button[kind="primary"] { color: #1d1915; background: #b39468; border-color: #b39468; }
    .stButton > button[kind="primary"]:hover { color: #171411; background: #c3a477; border-color: #c3a477; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ATT&CK report triage")
st.caption("Review a threat description, receive suggested ATT&CK tactics, and send uncertain results to a person.")

catalog = load_json(CATALOG_PATH)
model_path = available_model()
predictor = load_predictor(str(model_path)) if model_path else None
if predictor is None:
    st.error("The saved model could not be loaded. Run `python -m src.models.package_inference` to rebuild it.")

analyze_tab, batch_tab, notes_tab = st.tabs(["Analyze report", "Review batch", "Model notes"])
with analyze_tab:
    analyze_report(predictor, catalog)
with batch_tab:
    batch_queue(predictor)
with notes_tab:
    model_notes()

st.divider()
st.caption("Decision-support prototype · Predictions should be checked against the original intelligence context.")
