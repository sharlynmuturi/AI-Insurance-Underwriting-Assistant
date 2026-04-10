# AI Vehicle Insurance Underwriting Assistant

An end-to-end **AI-powered insurance underwriting system** that combines:

- Machine Learning for risk prediction
- Actuarial pricing models
- NLP-based policy understanding (TF-IDF + Sentence Transformers)
- Retrieval-Augmented Generation (RAG)
- LLM reasoning using Groq (LLaMA 3)

The system evaluates insurance applications, predicts risk, retrieves similar historical policies, and generates underwriting decisions with explanations.


1. Data Generation `generate_data.py`

A synthetic insurance dataset was created with realistic policy attributes:

- Driver demographics: age, experience, income band
- Vehicle details: type, age, value, airbags, tracking device
- Behavior: speeding score, annual mileage, previous claims
- Environment: region, traffic density
- Policy details: duration, risk score

Target variables:

- fraud_flag
- claim_occurred
- claim_amount


2. Exploratory Data Analysis (EDA) `notebooks\01_data_exploration.ipynb`

An extensive EDA was performed to ensure the dataset reflects a realistic insurance portfolio with appropriate variation in demographics, vehicles, and behavior.

Key insights explored:

- Distribution of claim amounts and premiums
- Relationship between policyholders' profile and claims
- Relationship between vehicle details and claims
- Risk differences across regions
- Premium adequacy analysis (Loss Ratio) to ensure pricing is fair and reasonable
- Correlation between previous claims and risk score


3. Risk Modeling using Machine Learning `notebooks\02_risk_modeling.ipynb`

Several ML models (Logistic Regression, Gradient Boosting Model, Random Forest and XGBoost) were trained to:

- Predict probability of claim occurrence
- Produce a calibrated risk score
- Support pricing decisions.

The XGBoost Model outperformed the other models and was selected for risk modeling.

4. Actuarial Pricing Engine

Premiums are calculated using expected loss pricing:

- expected_loss = model_predicted_risk × vehicle_value
> insurer’s anticipated claim cost
  
- technical_premium = expected_loss × (1 + expense_loading + profit_margin)
> premium ensuring operational costs are covered and profitability achieved

5. Policy Knowledge Base (RAG System)

Each policy is converted into structured text to create a **retrieval-ready knowledge base**:

Eg:

> “Age 35, sedan, Nairobi, 2 previous claims, tracking device installed…”


6. Retrieval System

Two retrieval approaches were implemented:

- TF-IDF - Converts policy text into sparse vectors and Uses word frequency weighting 
`notebooks\03_TF-IDF_rag.ipynb`

- Sentence Transformers - Uses `all-MiniLM-L6-v2` to convert policies into dense semantic embeddings. Captures meaning, not just keywords 
`notebooks\04_sentence_transformers_rag.ipynb`


7. RAG (Retrieval-Augmented Generation)

The system combines predicted risk score, recommended premium, similar historical policies and user query as inputs, then outputs (via Groq LLM):

 - Underwriting decision (Approve / Conditional / Decline)
 - Risk explanation
 - Pricing justification
 - Key risk drivers.

8. Streamlit Application

Interactive web app where users input driver details, vehicle details and policy parameters, outputs:

- Predicted risk score
- Recommended premium
- Similar policies from the knowledge base
- AI-generated underwriting explanation.


### How to Run

1. Install dependencies  
```bash
pip install -r requirements.txt  
```
2. Run Streamlit app  
```bash
streamlit run app.py
```


### Future Improvements

- Replace TF-IDF fully with vector database (FAISS / Pinecone)
- Fine-tune risk model with real claims data
- Add policy recommendation engine
- Add fraud detection module