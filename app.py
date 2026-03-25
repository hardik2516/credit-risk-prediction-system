import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Credit Risk System", layout="wide")

st.title("💳 Credit Risk Intelligence System")
st.caption("Machine Learning-based credit risk assessment system")

# ==============================
# LOAD MODELS
# ==============================

model_A = joblib.load("models/best_model.pkl")
threshold_A = joblib.load("models/threshold.pkl")

model_B = joblib.load("models/model_no_ext.pkl")
threshold_B = joblib.load("models/threshold_no_ext.pkl")

# ==============================
# SIDEBAR INPUTS
# ==============================

st.sidebar.header("📋 Customer Details")

income = st.sidebar.number_input("Annual Income (₹)", value=500000)
credit = st.sidebar.number_input("Loan Amount (₹)", value=200000)
annuity = st.sidebar.number_input("EMI (₹)", value=15000)

family_members = st.sidebar.number_input("Family Members", value=2)
children = st.sidebar.number_input("Children", value=0)

age = st.sidebar.number_input("Age", value=30)
employment_years = st.sidebar.number_input("Years Employed", value=5)

no_history = st.sidebar.checkbox("No Credit History")

if not no_history:
    ext1 = st.sidebar.slider("Credit Score 1", 0.0, 1.0, 0.5)
    ext2 = st.sidebar.slider("Credit Score 2", 0.0, 1.0, 0.5)
    ext3 = st.sidebar.slider("Credit Score 3", 0.0, 1.0, 0.5)

has_car = st.sidebar.selectbox("Owns Car?", ["Yes", "No"])
has_property = st.sidebar.selectbox("Owns Property?", ["Yes", "No"])

# ==============================
# CREATE DATA
# ==============================

data = pd.DataFrame({
    "AMT_INCOME_TOTAL": [income],
    "AMT_CREDIT": [credit],
    "AMT_ANNUITY": [annuity],
    "CNT_FAM_MEMBERS": [family_members],
    "CNT_CHILDREN": [children],
    "DAYS_BIRTH": [-age * 365],
    "DAYS_EMPLOYED": [-employment_years * 365],
    "FLAG_OWN_CAR": ['Y' if has_car == "Yes" else 'N'],
    "FLAG_OWN_REALTY": ['Y' if has_property == "Yes" else 'N']
})

# ==============================
# FEATURE ENGINEERING
# ==============================

data["EMI_INCOME_RATIO"] = data["AMT_ANNUITY"] / (data["AMT_INCOME_TOTAL"] + 1)
data["LOAN_INCOME_RATIO"] = data["AMT_CREDIT"] / (data["AMT_INCOME_TOTAL"] + 1)
data["TOTAL_PAYMENT_RATIO"] = (data["AMT_ANNUITY"] * 12) / (data["AMT_INCOME_TOTAL"] + 1)

data["INCOME_PER_PERSON"] = data["AMT_INCOME_TOTAL"] / (data["CNT_FAM_MEMBERS"] + 1)
data["EMPLOYMENT_AGE_RATIO"] = data["DAYS_EMPLOYED"] / (data["DAYS_BIRTH"] + 1)

# Save for UI
emi_ratio_ui = data["EMI_INCOME_RATIO"].values[0]

# ==============================
# EXT FEATURES
# ==============================

if not no_history:
    data["EXT_SOURCE_1"] = ext1
    data["EXT_SOURCE_2"] = ext2
    data["EXT_SOURCE_3"] = ext3
    data["EXT_SOURCE_MEAN"] = (ext1 + ext2 + ext3) / 3

# ==============================
# MODEL SELECTION
# ==============================

if no_history:
    model = model_B
    threshold = threshold_B
    st.warning("⚠️ Screening Mode (No Credit History)")
else:
    model = model_A
    threshold = threshold_A
    st.success("✅ Full Credit Evaluation Mode")

# ==============================
# ALIGN FEATURES
# ==============================

required_cols = model.named_steps['preprocessor'].feature_names_in_

for col in required_cols:
    if col not in data.columns:
        data[col] = 0

data = data[required_cols]

# ==============================
# PREDICTION
# ==============================

if st.button("🚀 Evaluate Application"):

    prob = model.predict_proba(data)[:, 1][0]

    st.markdown("---")

    # ==============================
    # METRICS
    # ==============================

    col1, col2, col3 = st.columns(3)

    col1.metric("Default Risk", f"{prob:.2f}")
    col2.metric("EMI Burden", f"{emi_ratio_ui:.2f}")
    col3.metric("Financial Stability", f"{1 - prob:.2f}")

    st.markdown("---")

    # ==============================
    # DECISION + RATING
    # ==============================

    if prob < 0.25:
        rating = "A"
        decision = "🟢 Loan Approved"
    elif prob < 0.45:
        rating = "B"
        decision = "🟢 Loan Approved"
    elif prob < threshold:
        rating = "C"
        decision = "🟡 Under Review"
    else:
        rating = "D"
        decision = "🔴 Loan Rejected"

    st.subheader(decision)
    st.write(f"**Risk Rating:** {rating}")

    # ==============================
    # KEY REASONS
    # ==============================

    st.markdown("### 🔍 Key Decision Factors")

    reasons = []

    if emi_ratio_ui > 0.30:
        reasons.append("High EMI compared to income")

    if credit / (income + 1) > 0.5:
        reasons.append("Loan amount is high relative to income")

    if not no_history and (ext1 + ext2 + ext3)/3 < 0.4:
        reasons.append("Low credit score")

    if employment_years < 2:
        reasons.append("Short employment history")

    if len(reasons) == 0:
        st.success("Strong financial profile with low repayment risk")
    else:
        for r in reasons[:3]:
            st.write(f"• {r}")

    # ==============================
    # FOOTNOTE
    # ==============================

    st.caption("""
    This assessment is based on financial indicators and predictive modeling.
    Final approval may require additional verification.
    """)
