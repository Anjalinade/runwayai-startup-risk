import streamlit as st
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
BACKEND_URL = "https://runwayai-startup-risk-1.onrender.com/predict"

st.set_page_config(
    page_title="RunwayAI – Risk Intelligence Dashboard",
    layout="wide"
)

st.title("🚀 RunwayAI – Startup Risk Intelligence")
st.write(
    "A decision intelligence system that explains **why** a startup is likely to fail or succeed."
)

st.divider()

# =========================
# INPUTS
# =========================
st.header("📥 Startup Profile")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", value=37.77)
    age_first_funding = st.number_input("Age at First Funding", value=2)
    relationships = st.number_input("Founder Relationships", value=10)
    funding_total = st.number_input("Total Funding (USD)", value=1_500_000)
    is_software = st.checkbox("Software Startup", value=True)
    has_angel = st.checkbox("Angel Investors", value=True)

with col2:
    longitude = st.number_input("Longitude", value=-122.41)
    funding_rounds = st.number_input("Funding Rounds", value=3)
    milestones = st.number_input("Milestones Achieved", value=4)
    has_vc = st.checkbox("VC Backing", value=True)
    is_top500 = st.checkbox("Top 500 Startup", value=False)

payload = {
    "latitude": latitude,
    "longitude": longitude,
    "age_first_funding_year": age_first_funding,
    "age_last_funding_year": 5,
    "age_first_milestone_year": 1,
    "age_last_milestone_year": 4,
    "relationships": relationships,
    "funding_rounds": funding_rounds,
    "funding_total_usd": funding_total,
    "milestones": milestones,

    "is_software": int(is_software),
    "is_web": 1,
    "is_enterprise": 1,

    "has_angel": int(has_angel),
    "has_VC": int(has_vc),
    "is_top500": int(is_top500),

    "is_CA": 1,
    "is_NY": 0,
    "is_MA": 0,
    "is_TX": 0,
    "is_otherstate": 0,
    "is_mobile": 0,
    "is_advertising": 0,
    "is_gamesvideo": 0,
    "is_ecommerce": 0,
    "is_biotech": 0,
    "is_consulting": 0,
    "is_othercategory": 0,
    "has_roundA": 1,
    "has_roundB": 0,
    "has_roundC": 0,
    "has_roundD": 0,
    "avg_participants": 3
}

# =========================
# PREDICTION
# =========================
st.divider()

if st.button("🔍 Analyze Startup Risk", use_container_width=True):

    response = requests.post(API_URL, json=payload)
    result = response.json()

    # -------------------------
    # RISK SUMMARY
    # -------------------------
    st.header("📊 Risk Assessment")

    risk_pct = round(result["failure_probability"] * 100, 2)
    risk_level = result["risk_level"]

    if risk_level == "High Risk":
        st.error(f"🔴 HIGH RISK — {risk_pct}% probability of failure")
    elif risk_level == "Medium Risk":
        st.warning(f"🟡 MEDIUM RISK — {risk_pct}% probability of failure")
    else:
        st.success(f"🟢 LOW RISK — {risk_pct}% probability of failure")

    st.divider()

    # -------------------------
    # EXPLANATION
    # -------------------------
    st.subheader("🧠 Why the model thinks this")

    col_risk, col_safe = st.columns(2)

    with col_risk:
        st.markdown("### ⚠️ Risk Contributors")
        for r in result["top_risk_factors"]:
            st.write("•", r)

    with col_safe:
        st.markdown("### 🛡️ Protective Contributors")
        for p in result["top_protective_factors"]:
            st.write("•", p)

    # -------------------------
    # RISK CONTRIBUTION CHART
    # -------------------------
    st.divider()
    st.subheader("📈 Risk Contribution Breakdown")

    contribution_data = []

    for r in result["top_risk_factors"]:
        contribution_data.append({"Factor": r, "Impact": 1})

    for p in result["top_protective_factors"]:
        contribution_data.append({"Factor": p, "Impact": -1})

    df = pd.DataFrame(contribution_data)

    st.bar_chart(
        df.set_index("Factor"),
        height=400
    )

    st.caption(
        "Positive bars increase failure risk. Negative bars reduce risk. "
        "Bar height represents relative influence on the decision."
    )
