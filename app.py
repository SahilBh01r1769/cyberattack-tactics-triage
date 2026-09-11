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


@st.cache_resource
def load_predictor(path: str) -> TacticPredictor:
    return TacticPredictor.from_artifact(path)


@st.cache_data
def load_json(path: str) -> list[dict]:
    return load_json_records(path)


def available_model() -> Path | None:
    return next((path for path in MODEL_CANDIDATES if path.exists()), None)


def render_tactic_map(catalog: list[dict], result: dict) -> None:
    predicted = {item["tactic"]: item["confidence"] for item in result["predictions"]}
    st.markdown("#### ATT&CK tactic map")
    for start in range(0, len(catalog), 5):
        columns = st.columns(5)
        for column, tactic in zip(columns, catalog[start : start + 5], strict=False):
            confidence = predicted.get(tactic["slug"])
            state = "active" if confidence is not None else "inactive"
            score = f"{confidence:.0%}" if confidence is not None else "—"
            tooltip = html.escape(tactic["description"])
            with column:
                st.markdown(
                    f'<div class="tactic-card {state}" title="{tooltip}">'
                    f'<span>{html.escape(tactic["external_id"])}</span>'
                    f'<strong>{html.escape(tactic["name"])}</strong>'
                    f'<b>{score}</b></div>',
                    unsafe_allow_html=True,
                )


def render_prediction(result: dict, predictor: TacticPredictor, text: str, catalog: list[dict]) -> None:
    auto_route = result["decision"] == "auto_route"
    state = "auto-route" if auto_route else "analyst review"
    icon = "✓" if auto_route else "!"
    st.markdown(
        f'<div class="decision {"accepted" if auto_route else "review"}">'
        f'<div class="decision-icon">{icon}</div><div><small>ROUTING DECISION</small>'
        f'<h3>{state.title()}</h3><p>{result["routing_confidence"]:.1%} routing confidence · '
        f'{result["routing_threshold"]:.0%} threshold</p></div></div>',
        unsafe_allow_html=True,
    )

    render_tactic_map(catalog, result)
    predictions = result["predictions"] or [result["top_candidate"]]
    if not result["predictions"]:
        st.caption("No label passed its learned threshold; showing the top candidate for analyst context.")

    st.markdown("#### What influenced this result")
    for prediction in predictions[:3]:
        tactic = prediction["tactic"]
        contributions = predictor.explain(text, tactic)
        with st.expander(
            f"{format_tactic(tactic)} · {prediction['confidence']:.1%}",
            expanded=tactic == predictions[0]["tactic"],
        ):
            if contributions:
                st.caption("Strongest positive TF-IDF contributions for this label")
                st.markdown(" ".join(f"`{item['feature']}`" for item in contributions))
            else:
                st.caption("No positive feature contributions were available for this input.")

    st.download_button(
        "Download result as JSON",
        json.dumps(result, indent=2),
        file_name="attack_triage_result.json",
        mime="application/json",
    )


def triage_workspace(predictor: TacticPredictor | None, threshold: float, catalog: list[dict]) -> None:
    st.subheader("Triage workspace")
    st.caption("Map a short threat report to one or more Enterprise ATT&CK tactics.")
    examples = load_json(EXAMPLES_PATH)
    left, right = st.columns([3, 1])
    with right:
        selected = st.selectbox(
            "ATT&CK-derived examples", range(len(examples)), format_func=lambda index: examples[index]["name"]
        )
        if st.button("Load example", width="stretch"):
            st.session_state.threat_text = examples[selected]["text"]
            st.session_state.example_source = examples[selected]
    with left:
        text = st.text_area(
            "Threat description",
            key="threat_text",
            height=160,
            placeholder="Paste a concise incident or threat-procedure description…",
        )
        classify = st.button("Analyze threat", type="primary", width="stretch")

    source = st.session_state.get("example_source")
    if source and source["text"] == text:
        st.caption(
            f"Example provenance: ATT&CK {source['technique_id']} · {source['technique_name']} · {source['source_name']}"
        )

    if classify:
        if predictor is None:
            st.error("The inference model is unavailable. Regenerate it with the commands in the sidebar.")
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
        render_prediction(result, predictor, text, catalog)


def batch_queue(predictor: TacticPredictor | None, threshold: float) -> None:
    st.subheader("Batch review queue")
    st.caption("Enrich up to 200 CTI records locally. Uploaded text is not sent to an external classification API.")
    upload = st.file_uploader("Upload CSV", type="csv", help="Include one column containing threat descriptions.")
    if upload is None:
        st.info("Upload a CSV to create an analyst review queue.")
        return
    if upload.size > 2_000_000:
        st.error("Keep the upload below 2 MB for this demo.")
        return
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    if len(frame) > 200:
        st.warning("Only the first 200 rows will be classified.")
        frame = frame.head(200)
    text_column = st.selectbox("Threat text column", frame.columns.tolist())
    if st.button("Classify queue", type="primary", disabled=predictor is None):
        try:
            st.session_state.batch_result = classify_frame(frame, text_column, predictor, threshold)
        except ValueError as exc:
            st.error(str(exc))

    result = st.session_state.get("batch_result")
    if result is not None:
        review_count = int(result["decision"].eq("analyst_review").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", len(result))
        c2.metric("Auto-routed", len(result) - review_count)
        c3.metric("Needs review", review_count)
        review_only = st.toggle("Show analyst-review cases only")
        display = result[result["decision"].eq("analyst_review")] if review_only else result
        st.dataframe(display, width="stretch", hide_index=True)
        st.download_button(
            "Download enriched CSV",
            safe_csv_bytes(result),
            file_name="attack_triage_queue.csv",
            mime="text/csv",
        )


def model_evidence() -> None:
    st.subheader("Model evidence")
    st.caption("Held-out test results use a conservative split grouped by ATT&CK source entity.")
    stats = json.loads(Path("artifacts/metrics/dataset_statistics.json").read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Examples", f"{stats['usable_samples']:,}")
    c2.metric("Techniques", stats["techniques"])
    c3.metric("Tactics", stats["tactics"])
    c4.metric("Multilabel", f"{stats['multilabel_fraction']:.1%}")

    comparison = pd.read_csv("artifacts/metrics/model_comparison.csv")
    display = comparison[["model", "macro_f1", "micro_f1", "samples_f1", "subset_accuracy"]].copy()
    display.columns = ["Model", "Macro F1", "Micro F1", "Samples F1", "Exact match"]
    st.markdown("#### Held-out model comparison")
    st.dataframe(
        display.style.format({column: "{:.3f}" for column in display.columns[1:]}).highlight_max(
            subset=display.columns[1:], color="#173f46"
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Linear SVM is the strongest model on the cleaned grouped-source split. Calibrated Logistic Regression "
        "powers live triage because it supports measured abstention and transparent local explanations."
    )

    stress = json.loads(Path("artifacts/metrics/technique_holdout_metrics.json").read_text())
    with st.expander("Unseen-technique stress test"):
        left, middle, right = st.columns(3)
        left.metric("Held-out techniques", stress["holdout_techniques"])
        middle.metric("Technique overlap", stress["technique_overlap"])
        right.metric("Macro F1", f"{stress['metrics']['macro_f1']:.3f}")
        st.caption(
            "This deliberately harder test holds out entire ATT&CK techniques. Its lower score shows that "
            "generalization to unseen behaviors remains a material limitation."
        )

    left, right = st.columns(2)
    with left:
        st.image("artifacts/figures/model_comparison.png", caption="Model progression")
    with right:
        st.image("artifacts/figures/confidence_coverage.png", caption="Confidence vs. automated coverage")
    with st.expander("Prior transformer run"):
        st.warning("These diagnostics predate the eight-example near-duplicate cleanup and are awaiting a clean-split rerun.")
        left, right = st.columns(2)
        left.image("artifacts/figures/transformer_training_history.png", caption="DistilBERT training history")
        right.image("artifacts/figures/transformer_per_label_comparison.png", caption="Per-label comparison")


st.set_page_config(page_title="ATT&CK Triage Workbench", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #07111d; color: #dce8f2; }
    [data-testid="stSidebar"] { background: #0b1724; border-right: 1px solid #1b3345; }
    h1, h2, h3, h4 { letter-spacing: -0.02em; }
    .hero { padding: 1.25rem 0 1rem; border-bottom: 1px solid #1b3345; margin-bottom: 1rem; }
    .hero small { color: #55d7d0; font-weight: 700; letter-spacing: .14em; }
    .hero h1 { margin: .2rem 0; font-size: 2.2rem; }
    .hero p { color: #8faabd; max-width: 760px; margin: 0; }
    .decision { display:flex; gap:1rem; align-items:center; padding:1rem 1.2rem; margin:1rem 0;
                background:#0c1b28; border:1px solid #244055; border-radius:10px; }
    .decision.accepted { border-left:4px solid #35c98b; }
    .decision.review { border-left:4px solid #f2ad4b; }
    .decision-icon { width:38px; height:38px; border-radius:50%; display:grid; place-items:center;
                     background:#152a38; font-size:1.2rem; font-weight:800; }
    .decision small, .tactic-card span { color:#7896aa; letter-spacing:.08em; font-size:.68rem; }
    .decision h3, .decision p { margin:0; }
    .tactic-card { min-height:92px; padding:.7rem; margin:.25rem 0; background:#0b1825;
                   border:1px solid #1a3041; border-radius:8px; display:flex; flex-direction:column; }
    .tactic-card strong { font-size:.82rem; margin:.28rem 0; line-height:1.15; }
    .tactic-card b { color:#5c788b; margin-top:auto; }
    .tactic-card.active { background:#0e2b32; border-color:#36bdb7; box-shadow:inset 0 0 0 1px #36bdb733; }
    .tactic-card.active b { color:#65e3db; }
    [data-testid="stMetric"] { background:#0b1825; border:1px solid #1a3041; padding:.8rem; border-radius:8px; }
    </style>
    <div class="hero"><small>ML-ASSISTED CYBER THREAT INTELLIGENCE</small>
    <h1>ATT&CK Triage Workbench</h1>
    <p>Map threat descriptions to Enterprise ATT&CK tactics, route uncertain cases to an analyst, and inspect the evidence behind the models.</p></div>
    """,
    unsafe_allow_html=True,
)

catalog = load_json(CATALOG_PATH)
model_path = available_model()
predictor = load_predictor(str(model_path)) if model_path else None

with st.sidebar:
    st.markdown("### Triage controls")
    threshold = st.slider(
        "Auto-route threshold",
        min_value=0.40,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="Higher values send more uncertain cases to analyst review.",
    )
    st.caption("At 0.70, held-out coverage was 78.0% with 87.8% micro F1 on accepted cases.")
    st.divider()
    st.markdown("**Live model**  ")
    st.caption("Calibrated TF-IDF + Logistic Regression")
    st.markdown("**Best verified model**  ")
    st.caption("Linear SVM · 0.773 macro F1")
    if predictor is None:
        st.warning("Inference artifact missing")
        st.code("python -m src.models.train_classical\npython -m src.models.calibrate", language="bash")

triage_tab, batch_tab, evidence_tab = st.tabs(["Triage workspace", "Batch queue", "Model evidence"])
with triage_tab:
    triage_workspace(predictor, threshold, catalog)
with batch_tab:
    batch_queue(predictor, threshold)
with evidence_tab:
    model_evidence()

st.caption("Decision-support prototype — not a replacement for analyst judgment or incident-response controls.")
