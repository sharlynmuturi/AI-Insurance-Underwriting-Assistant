# AI Vehicle Insurance Underwriting Assistant

An end-to-end **AI-powered insurance underwriting system** that combines:

- Machine Learning for risk prediction
- Actuarial pricing models
- NLP-based policy understanding (TF-IDF + Sentence Transformers)
- Retrieval-Augmented Generation (RAG)
- LLM reasoning using Groq (LLaMA 3)

The system evaluates insurance applications, predicts risk, retrieves similar historical policies, and generates underwriting decisions with explanations.

## Project Structure
### 1. Data Generation 

`generate_data.py`

Synthetic insurance dataset was generated using domain-driven assumptions on driver behavior, vehicle characteristics, geographic risk, and underwriting rules, with probabilistic modeling of claims frequency, severity, and pricing to simulate realistic insurance operations.

#### Policyholder Details
- Age is restricted to **21–70 years** (active driving population)
- Driving experience is approximated as - `years_experience = age - 18`
- Driver distribution reflects real-world exposure - Private (50%), Taxi (30%), Commercial (20%)
- Income bands skewed toward **low–middle income segments**
- License validity assumed high (**95% valid**)
- Taxi drivers more likely to belong to **SACCOs (informal transport structures)**

#### Vehicle Details 

- Vehicle type is **dependent on driver type** - Taxi(Sedan/SUV), Commercial (Pickup/Truck)
- Vehicle values are sampled from **realistic Kenyan market ranges**
- Depreciation modeled as **5% per year**
- Minimum vehicle value capped to avoid unrealistic values
- Majority vehicles assumed to be *Japan imports (60%)*

#### Vehicle Safety Details
- Airbags range from **0–7**
- Safety features distributed probabilistically - ABS (70%), ESC (50%)
- Tracking devices installed in ~40% of vehicles
- Safety features **reduce risk score**

#### Geographic Details
- Region distribution reflects population concentration - Nairobi (40%), Rural (25%), others smaller
- Urban areas have higher traffic density and crime rates
- Flood risk assigned to Mombasa & Kisumu (30% probability)

#### Policy Details

- Policy durations - 3, 6, 9, 12 months
- Comprehensive coverage more common (60%)
- Payment types - Annual (60%), Monthly (40%)

#### Driving Behavior
- Mileage varies by usage:
   - Private: 8K–20K km/year
   - Commercial: 20K–40K km/year
- Younger drivers more likely to have **high speeding scores**
- Driving intensity features - Night driving, weekend usage
- Harsh braking modeled via **Poisson distribution** with an average rate (λ = 5) to simulate driving behavior intensity.

#### Claims & Fraud
- Fraud rate assumed low (~5%)
- Time since last claim randomized if prior claims exist
- Previous claims follow **Poisson distribution (λ = 0.5)**.

Poisson distribution is used to model how many times something happens in a fixed period of time. Claim frequency is often modeled using Poisson since claims are random but with an average rate, rare and discrete.

For λ = 0.5, a driver has on average 0.5 claims per period roughly:

| Claims | Probability |
| --- | --- |
| 0 | ~60% |
| 1 | ~30% |
| 2 | ~7% |
| 3+ | very rare |

#### Risk Scoring Logic (Core Assumption)

Risk score is **rule-based and additive**, based on underwriting intuition:

##### Increases risk: 

- Young (<30) or older (>55) drivers
- Commercial usage
- Urban exposure (Nairobi)
- Short-term policies
- Old vehicles + low safety
- High mileage in high traffic areas
- More previous claims

##### Reduces risk:
- More airbags
- Presence of tracking device

Final risk score is clipped between **0 and 1**


#### Pricing

The dataset assumes a baseline pricing strategy set at approximately 5% of the vehicle value, then adjusted using a risk score to reflect the policyholder’s risk profile.
        
   `premium = base_premium * (1 + risk_score)`
        

#### Claims Generation Assumptions
- Claim probability driven by base rate (5%) + risk-adjusted component
- Claim severity placed at 10% - 70% of vehicle value
- Claim type logic:
    - Theft more likely in **high crime + no tracking**
    - Flood claims tied to **geography**
    - Otherwise - accident


### 2. Exploratory Data Analysis (EDA) 

`notebooks\01_data_exploration.ipynb`

An extensive EDA was performed to ensure the dataset reflects a realistic insurance portfolio with appropriate variation in demographics, vehicles, and behavior.

Key insights explored:

- Distribution of claim amounts and premiums
- Relationship between policyholders' profile and claims
- Relationship between vehicle details and claims
- Risk differences across regions
- Premium adequacy analysis (Loss Ratio) to ensure pricing is fair and reasonable
- Correlation between previous claims and risk score


### 3. Actuarial Modeling Framework

The system follows a **frequency-severity modeling approach**, commonly used in insurance pricing:

#### Frequency Model (Claim Occurrence)

`02_model_claim_frequency.ipynb`

Built a machine learning underwriting model to:

- Predict **claim probability (frequency)**
- Generate a **calibrated risk score**
- Support **pricing via expected loss + premium calculation**
- Feed into a **RAG-based AI underwriting assistant**

##### Data Preparation & Feature Engineering
- Removed **data leakage variables** (e.g. premium, claim amount, risk score)
- Created underwriting features:
    - **Safety Score** - captures protection level (airbags, tracking)
    - **Behavior Score** - captures driving risk (mileage, claims, speeding)
    - **Risk buckets** - age bands, vehicle age bands
    - **Exposure indicator** - high mileage flag
- Applied **one-hot encoding** for categorical variables
- Used **correlation checks** to confirm no leakage remained

##### Class Imbalance Handling

Only ~8% of policies resulted in claims.

Different strategies were used depending on the model:

- Logistic Regression - `class_weight="balanced"`
- Random Forest - SMOTE oversampling
- XGBoost - `scale_pos_weight`


##### Models Trained & Insights

###### Logistic Regression (Baseline)
-  ROC-AUC: ~0.58 (best overall ranking performance)
- Captured key drivers:
    - Higher risk: urban region, behavior score
    - Lower risk: safer driving, longer policies
- Limitation: Poor separation of high-risk segments

###### Gradient Boosting
- Underperformed (recall ≈ 0)
- Failed to identify claim cases effectively  

**Rejected**

###### Random Forest (with SMOTE)

- Improved balance vs GBM
- Produced **better risk segmentation (monotonic bands)**
- Still low recall for claims  

**Moderately useful**

###### XGBoost (Final Model)
- Best **recall (~68%)** - captures most risky policies
- Risk bands show **clear monotonic relationship** (Higher predicted risk - higher actual claim rate)
- Feature importance aligned with real-world logic:
    - Urban exposure (Nairobi)
    - Driving behavior
    - Vehicle type & usage
    - Policy duration

**Selected as final underwriting model**

#### Severity Models (Claim Cost Given Claim)

`03_model_claim_severity.ipynb`

Only claim-positive records were used.

Two actuarial regression approaches were tested:

##### Gamma Regression (GLM)
- Models **positive continuous claim amounts**
- Assumes Gamma distribution of severity
- Performs well for smooth, skewed financial data

##### Tweedie Regression (GLM)
- Models **compound distribution (frequency + severity behavior)**
- Naturally suited for insurance claims (many zeros + positive skewed losses)

##### Model Comparison

| Model | Performance | Observation |
| --- | --- | --- |
| Gamma Regression | Higher MAE / RMSE | Underfit complex non-linear patterns |
| Tweedie Regression | Lower MAE / RMSE | Better overall predictive accuracy |

**Tweedie Regression was selected as the final severity model**

### 4. Pricing Engine (Actuarial Approach)

The final pricing system follows a **two-stage actuarial model**:

#### Step 1: Expected Loss (Pure Premium)

`Expected Loss = Frequency × Severity`

Where:
- Frequency = probability of claim (XGBoost)
- Severity = predicted claim size (Tweedie model)

#### Step 2: Technical Premium

`Technical Premium = Expected Loss × (1 + Expense Loading + Profit Margin)`

Where:

- Expense Loading = operational + acquisition costs (assumed 30%)
- Profit Margin = insurer return target (assumed 10%)

The system produces:

*   Individual policy **claim probability**
*   Expected **claim severity**
*   Combined **expected loss**
*   Final **technical premium per policy**

### 4. RAG System 

`notebooks\03_TF-IDF_rag.ipynb`

`notebooks\04_sentence_transformers_rag.ipynb`

#### Policy Knowledge Base

Each policy is converted into structured text to create a **retrieval-ready knowledge base**:

Eg:

> “Age 35, sedan, Nairobi, 2 previous claims, tracking device installed…”


#### Retrieval System

Two retrieval approaches were implemented:

- TF-IDF - Converts policy text into sparse vectors and Uses word frequency weighting 
- Sentence Transformers - Uses `all-MiniLM-L6-v2` to convert policies into dense semantic embeddings. Captures meaning, not just keywords 

#### RAG (Retrieval-Augmented Generation)

The system combines predicted risk score, recommended premium, similar historical policies and user query as inputs, then outputs (via Groq LLM):

 - Underwriting decision (Approve / Conditional / Decline)
 - Risk explanation
 - Pricing justification
 - Key risk drivers.

### 5. Streamlit Application

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