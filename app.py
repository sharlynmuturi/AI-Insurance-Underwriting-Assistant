import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = Path(__file__).parent

tweedie_model_path = BASE_DIR / "artifacts" / "tweedie_model.pkl"
xgb_model_path = BASE_DIR / "artifacts" / "xgb_model.pkl"

tweedie_feature_columns_path = BASE_DIR / "artifacts" / "tweedie_feature_columns.pkl"
xgb_feature_columns_path = BASE_DIR / "artifacts" / "xgb_feature_columns.pkl"

dataset_path = BASE_DIR / "artifacts" / "insurance_dataset.csv"

@st.cache_data
def load_resources():
    tweedie_model = joblib.load(tweedie_model_path)
    tweedie_feature_columns = joblib.load(tweedie_feature_columns_path)

    xgb_model = joblib.load(xgb_model_path)
    xgb_feature_columns = joblib.load(xgb_feature_columns_path)

    # Load dataset for KB
    df = pd.read_csv(dataset_path)

    # Create text representation if not exists
    if "policy_text" not in df.columns:
        kb_features = ["age", "driver_type", "vehicle_type", "vehicle_age",
                       "annual_mileage", "previous_claims", "risk_score",
                       "airbags", "tracking_device", "region", "policy_duration"]

        def policy_to_text(row):
            return (
                f"Policyholder: age {row['age']}, driver_type {row['driver_type']}, "
                f"vehicle_type {row['vehicle_type']} ({row['vehicle_age']} yrs old), "
                f"annual_mileage {row['annual_mileage']}, previous_claims {row['previous_claims']}, "
                f"safety: airbags {row['airbags']}, tracking_device {row['tracking_device']}, "
                f"region {row['region']}, policy_duration {row['policy_duration']} months, "
                f"risk_score {row['risk_score']:.2f}."
            )
        df["policy_text"] = df.apply(policy_to_text, axis=1)

    # Vectorize for retrieval
    vectorizer = TfidfVectorizer(max_features=5000)
    policy_vectors = vectorizer.fit_transform(df["policy_text"])

    return tweedie_model, xgb_model, tweedie_feature_columns, xgb_feature_columns, df, vectorizer, policy_vectors

tweedie_model, xgb_model, tweedie_feature_columns, xgb_feature_columns, df, vectorizer, policy_vectors = load_resources()


client = ChatGroq(
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

def ask_groq(prompt):
    response = client.invoke([
        {"role": "system", "content": "You are an expert insurance underwriter."},
        {"role": "user", "content": prompt}
    ])
    return response.content


def retrieve_similar_policies(query_text, top_k=5):
    query_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, policy_vectors).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return df.iloc[top_idx][["policy_text", "risk_score", "claim_occurred"]], sims[top_idx]


def predict_risk_from_query(query_dict):
    input_df = pd.DataFrame([query_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=xgb_feature_columns, fill_value=0)
    return xgb_model.predict_proba(input_df)[:, 1][0]


def calculate_premium(predicted_risk, vehicle_value,
                      expense_loading=0.3, profit_margin=0.1):
    expected_loss = predicted_risk * vehicle_value
    premium = expected_loss * (1 + expense_loading + profit_margin)
    premium = max(premium, 5000)
    return expected_loss, premium


def predict_pure_premium(input_dict):
    input_df = pd.DataFrame([input_dict])
    
    input_df = pd.get_dummies(input_df)
    # Align with training columns
    input_df = input_df.reindex(columns=tweedie_feature_columns, fill_value=0)
    
    # Predict expected loss (pure premium)
    pure_premium = tweedie_model.predict(input_df)[0]
    
    return pure_premium

def calculate_final_premium(pure_premium, expense_loading=0.3, profit_margin=0.1):
    return pure_premium * (1 + expense_loading + profit_margin)

def build_prompt(user_question, retrieved_policies, predicted_risk, pure_premium, premium):
    context = "\n".join(retrieved_policies["policy_text"].tolist())
    prompt = f"""
You are an insurance underwriting assistant.

You MUST respond in the exact format below. Keep it concise.

FORMAT:

Underwriting Decision: (Approve / Conditional / Decline)

Risk Summary:
- Predicted Risk Score: {predicted_risk:.2f}
- Key Drivers: list in short bullet points

Pricing:
- Pure Premium (Expected Loss): {pure_premium:.0f}
- Recommended Premium: {premium:.0f}

Justification:
- 3–5 short sentences max

Similar Cases Insight:
- 2 bullet points max

DO NOT:
- Do not repeat calculations
- Do not explain methodology
- Do not be verbose
- No paragraphs longer than 3 lines

SIMILAR POLICIES:
{context}

QUESTION:
{user_question}

ANSWER:
"""
    return prompt


# UI
st.title("AI Vehicle Insurance Underwriter")

st.header("Enter Policy Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Driver Info")

    age = st.number_input("Age", 18, 100, 30)
    driver_type = st.selectbox("Driver Type", ["private", "taxi", "commercial"])
    previous_claims = st.number_input("Previous Claims", 0, value=0)
    speeding_score = st.selectbox("Speeding Score", ["low", "medium", "high"])

with col2:
    st.subheader("Vehicle Info")

    vehicle_type = st.selectbox("Vehicle Type", ["sedan", "SUV", "pickup", "truck"])
    vehicle_age = st.number_input("Vehicle Age (years)", 0, 20, 3)
    vehicle_value = st.number_input("Vehicle Value (KES)", 50000, value=120000)
    annual_mileage = st.number_input("Annual Mileage (km)", 0, value=15000)

with col3:
    st.subheader("Policy & Safety")

    airbags = st.number_input("Number of Airbags", 0, value=4)
    tracking_device = st.selectbox("Tracking Device Installed?", [1, 0])
    region = st.selectbox("Region", ["Nairobi", "Mombasa", "Kisumu", "Rural"])
    policy_duration = st.selectbox("Policy Duration (months)", [3, 6, 9, 12])

query_features = {
    "age": age,
    "driver_type": driver_type,
    "vehicle_age": vehicle_age,
    "vehicle_type": vehicle_type,
    "vehicle_value": vehicle_value,
    "annual_mileage": annual_mileage,
    "previous_claims": previous_claims,
    "airbags": airbags,
    "tracking_device": tracking_device,
    "region": region,
    "policy_duration": policy_duration,
    "speeding_score": speeding_score
}

user_question = f"Assess risk and recommend premium for a policy duration of {policy_duration} months for a {age}-year-old {driver_type} driver with a {vehicle_age}-year-old {vehicle_type} valued at {vehicle_value} with {annual_mileage} km/year mileage, {airbags} number of airbags and {tracking_device} tracking devices in {region}, previous claims: {previous_claims}"

# Action button
if st.button("Evaluate Policy"):
    with st.spinner("Evaluating..."):
        # Retrieve
        retrieved_policies, sims = retrieve_similar_policies(user_question)

        pure_premium = predict_pure_premium(query_features)
        premium = calculate_final_premium(pure_premium)

        predicted_risk = predict_risk_from_query(query_features)

        # Build prompt
        prompt = build_prompt(user_question, retrieved_policies, predicted_risk, pure_premium, premium)

        # Ask LLM
        answer = ask_groq(prompt)

    st.subheader("Predicted Risk & Premium")

    colA, colB, colC = st.columns(3)

    colA.metric("Predicted Risk Score", f"{predicted_risk:.2f}")
    colB.metric("Pure Premium (Expected Loss)", f"KES {pure_premium:,.0f}")
    colC.metric("Loaded Premium", f"KES {premium:,.0f}")

    st.subheader("AI Underwriter Decision")
    st.write(answer)

    st.subheader("Similar Policies (Knowledge Based)")
    st.dataframe(retrieved_policies)