import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_data
def load_resources():
    # Load model
    model = joblib.load("xgb_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    # Load dataset for KB
    df = pd.read_csv("insurance_dataset.csv")

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

    return model, feature_columns, df, vectorizer, policy_vectors

model, feature_columns, df, vectorizer, policy_vectors = load_resources()


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
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return model.predict_proba(input_df)[:, 1][0]

def calculate_premium(predicted_risk, vehicle_value,
                      expense_loading=0.3, profit_margin=0.1):
    expected_loss = predicted_risk * vehicle_value
    premium = expected_loss * (1 + expense_loading + profit_margin)
    premium = max(premium, 5000)
    return expected_loss, premium

def build_prompt(user_question, retrieved_policies, predicted_risk, premium):
    context = "\n".join(retrieved_policies["policy_text"].tolist())
    prompt = f"""
You are an AI insurance underwriting assistant.

PREDICTED RISK SCORE: {predicted_risk:.2f}
RECOMMENDED PREMIUM: {premium:.0f}

SIMILAR POLICIES:
{context}

QUESTION:
{user_question}

INSTRUCTIONS:
- Provide underwriting decision (Approve / Conditional / Decline)
- Explain reasoning using risk score and similar cases
- Justify the recommended premium
- Highlight key risk drivers

ANSWER:
"""
    return prompt


# UI
st.title("AI Vehicle Insurance Underwriter")

st.header("Enter Policy Details")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
driver_type = st.selectbox("Driver Type", ["private", "taxi", "commercial"])
vehicle_type = st.selectbox("Vehicle Type", ["sedan", "SUV", "pickup", "truck"])
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=20, value=3)
vehicle_value = st.number_input("Vehicle Value (KES)", min_value=50000, value=120000)
annual_mileage = st.number_input("Annual Mileage (km)", min_value=0, value=15000)
previous_claims = st.number_input("Previous Claims", min_value=0, value=0)
airbags = st.number_input("Number of Airbags", min_value=0, value=4)
tracking_device = st.selectbox("Tracking Device Installed?", [1,0])
region = st.selectbox("Region", ["Nairobi", "Mombasa", "Kisumu", "Rural"])
policy_duration = st.selectbox("Policy Duration (months)", [3,6,9,12])
speeding_score = st.selectbox("Speeding Score", ["low", "medium", "high"])

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

user_question = f"Assess risk and recommend premium for a {age}-year-old {driver_type} driver with a {vehicle_age}-year-old {vehicle_type}, with {airbags} number of airbags and {tracking_device} tracking devices in {region}, previous claims: {previous_claims}"

# Action button
if st.button("Evaluate Policy"):
    with st.spinner("Evaluating..."):
        # Retrieve
        retrieved_policies, sims = retrieve_similar_policies(user_question)

        # Predict
        predicted_risk = predict_risk_from_query(query_features)

        # Premium
        expected_loss, premium = calculate_premium(predicted_risk, vehicle_value)

        # Build prompt
        prompt = build_prompt(user_question, retrieved_policies, predicted_risk, premium)

        # Ask LLM
        answer = ask_groq(prompt)

    st.subheader("Predicted Risk & Premium")
    st.write(f"Predicted Risk Score: {predicted_risk:.2f}")
    st.write(f"Recommended Premium: KES {premium:,.0f}")

    st.subheader("AI Underwriter Explanation")
    st.write(answer)

    st.subheader("Similar Policies (Knowledge Based)")
    st.dataframe(retrieved_policies)
