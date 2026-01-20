import streamlit as st
import requests

# ======================================
# Page Config
# ======================================
st.set_page_config(
    page_title="RunwayAI",
    layout="centered"
)

st.title("🚀 RunwayAI – Startup Failure Risk Predictor")
st.write(
    "This tool predicts the **risk of startup failure** and explains "
    "**which factors increased or reduced that risk**."
)

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.divider()

# ======================================
# Human-Readable Feature Names
# ======================================
FEATURE_NAMES = {
    "age_first_funding_year": "Early funding access",
    "age_last_funding_year": "Late funding dependency",
    "age_first_milestone_year": "Early milestones",
    "age_last_milestone_year": "Delayed milestones",
    "relationships": "Founder network strength",
    "funding_rounds": "Number of funding rounds",
    "funding_total_usd": "Total funding raised",
    "milestones": "Business milestones",
    "avg_participants": "Investor participation",
    "is_software": "Software startup",
    "is_web": "Web-based product",
    "is_mobile": "Mobile product",
    "is_enterprise": "Enterprise focus",
    "has_VC": "VC backing",
    "has_angel": "Angel funding",
    "is_top500": "Top-500 startup"
}

# ======================================
# Inputs
# ======================================
st.subheader("📊 Startup Metrics")

latitude = st.number_input("Latitude", value=37.77)
longitude = st.number_input("Longitude", value=-122.41)

age_first_funding_year = st.number_input("Age at First Funding (years)", value=2)
age_last_funding_year = st.number_input("Age at Last Funding (years)", value=5)

age_first_milestone_year = st.number_input("Age at First Milestone (years)", value=1)
age_last_milestone_year = st.number_input("Age at Last Milestone (years)", value=4)

relationships = st.number_input("Founder Relationships", value=10)
funding_rounds = st.number_input("Funding Rounds", value=3)
funding_total_usd = st.number_input("Total Funding (USD)", value=1_500_000)
milestones = st.number_input("Milestones Achieved", value=4)
avg_participants = st.number_input("Avg Investors per Round", value=3)

st.divider()

st.subheader("🏷 Category & Funding")

is_software = st.checkbox("Software", value=True)
is_web = st.checkbox("Web", value=True)
is_mobile = st.checkbox("Mobile")
is_enterprise = st.checkbox("Enterprise", value=True)

has_VC = st.checkbox("VC Funding", value=True)
has_angel = st.checkbox("Angel Funding", value=True)
is_top500 = st.checkbox("Top-500 Startup")

st.divider()

# ======================================
# Payload (FULL – schema safe)
# ======================================
payload = {
    "latitude": latitude,
    "longitude": longitude,

    "age_first_funding_year": age_first_funding_year,
    "age_last_funding_year": age_last_funding_year,
    "age_first_milestone_year": age_first_milestone_year,
    "age_last_milestone_year": age_last_milestone_year,

    "relationships": relationships,
    "funding_rounds": funding_rounds,
    "funding_total_usd": funding_total_usd,
    "milestones": milestones,
    "avg_participants": avg_participants,

    "is_CA": 0,
    "is_NY": 0,
    "is_MA": 0,
    "is_TX": 0,
    "is_otherstate": 1,

    "is_software": int(is_software),
    "is_web": int(is_web),
    "is_mobile": int(is_mobile),
    "is_enterprise": int(is_enterprise),
    "is_advertising": 0,
    "is_gamesvideo": 0,
    "is_ecommerce": 0,
    "is_biotech": 0,
    "is_consulting": 0,
    "is_othercategory": 0,

    "has_VC": int(has_VC),
    "has_angel": int(has_angel),
    "has_roundA": 1,
    "has_roundB": 0,
    "has_roundC": 0,
    "has_roundD": 0,

    "is_top500": int(is_top500),
}

# ======================================
# Predict
# ======================================
if st.button("🔮 Predict Failure Risk"):
    with st.spinner("Analyzing startup risk..."):
        response = requests.post(BACKEND_URL, json=payload)

    if response.status_code != 200:
        st.error("Backend rejected the request.")
        st.stop()

    result = response.json()

    prob = result["failure_probability"]
    risk = result["risk_level"]

    st.subheader("📊 Risk Assessment")

    if risk == "Low Risk":
        st.success(f"🟢 Low Risk — {prob:.2%}")
    elif risk == "Medium Risk":
        st.warning(f"🟡 Medium Risk — {prob:.2%}")
    else:
        st.error(f"🔴 High Risk — {prob:.2%}")

    st.progress(prob)

    # ======================================
    # TOP RISK FACTORS
    # ======================================
    st.subheader("🔴 Top Risk Drivers")

    for f in result["top_risk_factors"]:
        if f in FEATURE_NAMES:
            st.write(f"• {FEATURE_NAMES[f]}")

    st.subheader("🟢 Protective Factors")

    for f in result["positive_signals"]:
        if f in FEATURE_NAMES:
            st.write(f"• {FEATURE_NAMES[f]}")
