from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.inference.predict import TacticPredictor


MODEL_PATH = Path("artifacts/models/confidence_logistic_regression.joblib")


@st.cache_resource
def load_predictor() -> TacticPredictor:
    return TacticPredictor.from_artifact(MODEL_PATH)


st.set_page_config(page_title="ATT&CK Tactic Triage", page_icon="🛡️", layout="centered")
st.title("ATT&CK tactic triage")
st.caption("Multilabel classification of short cyber-threat procedure descriptions")

if not MODEL_PATH.exists():
    st.error("The calibrated model has not been generated on this machine.")
    st.code("python -m src.models.train_classical\npython -m src.models.calibrate", language="bash")
    st.stop()

description = st.text_area(
    "Threat description",
    height=150,
    placeholder="Example: The adversary executed a PowerShell command to download a payload.",
)

if st.button("Classify", type="primary"):
    if not description.strip():
        st.warning("Enter a threat description first.")
    else:
        result = load_predictor().predict(description)
        if result["decision"] == "auto_route":
            st.success(f"Auto-route · confidence {result['routing_confidence']:.1%}")
        else:
            st.warning(f"Analyst review recommended · confidence {result['routing_confidence']:.1%}")

        st.subheader("Predicted tactics")
        if result["predictions"]:
            for prediction in result["predictions"]:
                label = prediction["tactic"].replace("-", " ").title()
                st.write(f"**{label}** — {prediction['confidence']:.1%}")
        else:
            candidate = result["top_candidate"]
            label = candidate["tactic"].replace("-", " ").title()
            st.write(f"No tactic passed its label threshold. Top candidate: **{label}** ({candidate['confidence']:.1%})")

        st.caption(
            f"Routing threshold: {result['routing_threshold']:.0%}. "
            "Low-confidence cases are intentionally deferred rather than forced into automatic routing."
        )
