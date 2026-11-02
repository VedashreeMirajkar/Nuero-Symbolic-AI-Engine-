"""Streamlit dashboard.  Run:  streamlit run app.py"""
import streamlit as st

from nsre.config import MODEL_PATH, RULE_THRESHOLDS
from nsre.data import generate_dataset, save_dataset
from nsre.engine import ReasoningEngine
from nsre.hybrid import PolicyConfig
from nsre.model import load_bundle, save_bundle, train_model

st.set_page_config(page_title="Neuro-Symbolic Reasoning Engine", layout="wide")


@st.cache_resource
def get_bundle():
    if not MODEL_PATH.exists():
        df = generate_dataset()
        save_dataset(df)
        save_bundle(train_model(df))
    return load_bundle()


EXAMPLES = {
    "Custom": None,
    "Clearly low risk": dict(annual_income=90000, debt_to_income=0.20, credit_history_years=14, missed_payments=0, savings_ratio=0.35),
    "Agreement: high risk": dict(annual_income=30000, debt_to_income=0.65, credit_history_years=3, missed_payments=5, savings_ratio=0.02),
    "Likely model/rule conflict": dict(annual_income=120000, debt_to_income=0.47, credit_history_years=15, missed_payments=3, savings_ratio=0.40),
    "Invalid input": dict(annual_income=50000, debt_to_income=2.5, credit_history_years=5, missed_payments=1, savings_ratio=0.1),
}

st.title("Neuro-Symbolic Reasoning Engine")
st.warning("Educational prototype on SYNTHETIC data. Not a validated credit-scoring or lending system. "
           "All thresholds are illustrative.")

with st.sidebar:
    st.header("Applicant")
    ex = st.selectbox("Load example", list(EXAMPLES))
    d = EXAMPLES[ex] or dict(annual_income=60000, debt_to_income=0.30, credit_history_years=8, missed_payments=0, savings_ratio=0.20)
    rec = {
        "annual_income": st.number_input("Annual income", value=float(d["annual_income"]), step=1000.0, key=f"i{ex}"),
        "debt_to_income": st.number_input("Debt-to-income ratio", value=float(d["debt_to_income"]), step=0.01, format="%.3f", key=f"d{ex}"),
        "credit_history_years": st.number_input("Credit history (years)", value=float(d["credit_history_years"]), step=0.5, key=f"h{ex}"),
        "missed_payments": st.number_input("Missed payments", value=float(d["missed_payments"]), step=1.0, key=f"m{ex}"),
        "savings_ratio": st.number_input("Savings ratio", value=float(d["savings_ratio"]), step=0.01, format="%.3f", key=f"s{ex}"),
    }
    st.header("Hybrid policy")
    med = st.slider("Medium-floor score", 1, 6, 3)
    high = st.slider("High-floor score", 2, 9, 5)
    res = st.slider("Model confidence that resists override", 0.5, 1.0, 0.90, 0.01)

engine = ReasoningEngine(get_bundle(), policy=PolicyConfig(medium_floor_score=med, high_floor_score=high, confidence_resistance=res))
a = engine.assess(rec)

if not a.valid:
    st.error("INVALID INPUT - no risk assessment produced")
    for v in a.validation.violations:
        st.write(f"**{v.constraint_id}** - {v.message}")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Neural prediction", a.neural.label.upper(), f"{a.neural.confidence:.0%} confidence")
    c2.metric("Symbolic floor", a.decision.symbolic_floor.upper(), f"score {a.decision.symbolic_score}")
    c3.metric("Hybrid result", a.decision.hybrid_risk.upper(), a.decision.decision_source)
    (st.success if a.decision.agree else st.warning)(
        f"Neural vs rules: {'AGREE' if a.decision.agree else 'DISAGREE'} ({a.decision.agreement_type})")
    if a.decision.needs_review:
        st.info("Flagged for manual review.")
    left, right = st.columns(2)
    with left:
        st.subheader("Neural class probabilities")
        st.bar_chart(a.neural.probabilities)
    with right:
        st.subheader("Rules")
        for r in a.findings:
            st.write(("FIRED " if r.fired else "not fired ") + f"**{r.rule_id}** {r.name} [{r.severity}]")
            if r.fired:
                st.caption(f"{r.condition} -> {r.explanation}")

st.subheader("Explanation")
st.code(a.explanation, language="text")
