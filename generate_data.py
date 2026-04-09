import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N = 50000  # number of policies


def random_choice(options, probs):
    return np.random.choice(options, p=probs)

def generate_dates(n):
    start = datetime(2022, 1, 1)
    return [start + timedelta(days=np.random.randint(0, 365)) for _ in range(n)]


# 1. Policyholder Info
ages = np.random.randint(21, 70, N)
driver_type = np.random.choice(["private", "taxi", "commercial"], size=N, p=[0.5, 0.3, 0.2])
years_experience = np.clip(ages - 18, 1, None)
gender = np.random.choice(["male", "female"], N, p=[0.7, 0.3])
marital_status = np.random.choice(["single", "married"], N, p=[0.5, 0.5])
income_band = np.random.choice(["low", "middle", "high"], size=N, p=[0.5, 0.35, 0.15])
ntsa_license_valid = np.random.choice([1, 0], N, p=[0.95, 0.05])

sacco_member = [
    1 if dt in ["taxi", "boda_boda"] and np.random.rand() < 0.6 else 0
    for dt in driver_type
]


# 2. Vehicle Info
vehicle_age = np.random.randint(0, 15, N)

vehicle_type = []

for dt in driver_type:
    if dt == "taxi":
        vehicle_type.append(np.random.choice(["sedan", "SUV"], p=[0.7, 0.3]))
    elif dt == "commercial":
        vehicle_type.append(np.random.choice(["pickup", "truck"], p=[0.6, 0.4]))
    else:
        vehicle_type.append(np.random.choice(["sedan", "SUV"], p=[0.6, 0.4]))

vehicle_type = np.array(vehicle_type)

fuel_type = np.random.choice(["petrol", "diesel"], N, p=[0.7, 0.3])
transmission = np.random.choice(["manual", "automatic"], N, p=[0.6, 0.4])
vehicle_origin = np.random.choice(["japan_import", "local", "new"], N, p=[0.6, 0.25, 0.15])

usage_type = [
    "ride_hailing" if dt == "taxi" else dt
    for dt in driver_type
]

vehicle_base_values = {
    "sedan": (800000, 1800000),
    "SUV": (1500000, 4000000),
    "pickup": (1200000, 3500000),
    "truck": (2500000, 7000000)
}

vehicle_value = []

for vt in vehicle_type:
    low, high = vehicle_base_values[vt]
    value = np.random.uniform(low, high)
    vehicle_value.append(value)

vehicle_value = np.array(vehicle_value)

# Adding depreciation
vehicle_value = vehicle_value * (1 - (vehicle_age * 0.05))
vehicle_value = np.clip(vehicle_value, 50000, None)

# 3. Safety Features
airbags = np.random.randint(0, 8, N)
abs = np.random.choice([1, 0], N, p=[0.7, 0.3])
esc = np.random.choice([1, 0], N, p=[0.5, 0.5])
tracking_device = np.random.choice([1, 0], N, p=[0.4, 0.6])
anti_theft_device = np.random.choice([1, 0], N, p=[0.5, 0.5])


# 4. Geography
region = np.random.choice(["Nairobi", "Mombasa", "Kisumu", "Rural"], N, p=[0.4, 0.2, 0.15, 0.25])

area_type = [
    "urban" if r == "Nairobi" else
    "peri_urban" if r in ["Mombasa", "Kisumu"] else
    "rural"
    for r in region
]

traffic_density = [
    "high" if r == "Nairobi" else
    "medium" if r in ["Mombasa", "Kisumu"] else
    "low"
    for r in region
]

crime_rate_area = [
    "high" if r == "Nairobi" else
    "medium" if r in ["Mombasa"] else
    "low"
    for r in region
]

flood_risk_area = [
    1 if r in ["Mombasa", "Kisumu"] and np.random.rand() < 0.3 else 0
    for r in region
]


# 5. Policy Details
policy_duration = np.random.choice([3, 6, 9, 12], N, p=[0.2, 0.3, 0.2, 0.3])
coverage_type = np.random.choice(["third_party", "comprehensive"], N, p=[0.4, 0.6])
policy_start_date = generate_dates(N)
payment_type = np.random.choice(["annual", "monthly"], N, p=[0.6, 0.4])


# 6. Driving Behavior
annual_mileage = []

for dt in driver_type:
    if dt == "private":
        annual_mileage.append(np.random.randint(8000, 20000))
    else:
        annual_mileage.append(np.random.randint(20000, 40000))

annual_mileage = np.array(annual_mileage)

avg_daily_distance = annual_mileage / 365

night_driving_ratio = np.random.uniform(0.1, 0.5, N)
weekend_usage_ratio = np.random.uniform(0.2, 0.6, N)

# Speeding influenced by age
speeding_score = []
for age in ages:
    if age < 30:
        speeding_score.append("high")
    elif age < 50:
        speeding_score.append("medium")
    else:
        speeding_score.append("low")

harsh_braking = np.random.poisson(5, N)


# 7. Claims History
previous_claims = np.random.poisson(0.5, N)

last_claim_years = [
    np.random.randint(1, 5) if pc > 0 else 0
    for pc in previous_claims
]

fraud_flag = np.random.choice([1, 0], N, p=[0.05, 0.95])


# 8. CLAIM PROBABILITY LOGIC
risk_score = []

for i in range(N):
    score = 0

    # Age risk
    if ages[i] < 30:
        score += 0.2
    elif ages[i] > 55:
        score += 0.15

    # Driver type risk
    if driver_type[i] in ["commercial"]:
        score += 0.25

    # Urban risk
    if region[i] == "Nairobi":
        score += 0.2

    # Short policy risk
    if policy_duration[i] <= 6:
        score += 0.2

    # old car + no safety
    if vehicle_age[i] > 10 and airbags[i] < 2:
        score += 0.2

    # high mileage + urban
    if annual_mileage[i] > 30000 and traffic_density[i] == "high":
        score += 0.2

    # Previous claims
    score += previous_claims[i] * 0.1

    # Safety reduces risk
    score -= airbags[i] * 0.01
    if tracking_device[i] == 1:
        score -= 0.05

    risk_score.append(score)

risk_score = np.clip(risk_score, 0, 1)
risk_score = np.array(risk_score)  # convert list to numpy array

base_premium = 0.05 * vehicle_value
premium = base_premium * (1 + risk_score)

claim_prob = 0.05 + (risk_score * 0.10)

claim_occurred = (np.random.rand(N) < claim_prob).astype(int)

# Claim amount
claim_amount = []

for i in range(N):
    if claim_occurred[i] == 1:
        severity = np.random.uniform(0.1, 0.7)  # % of vehicle value
        claim_amount.append(vehicle_value[i] * severity)
    else:
        claim_amount.append(0)

claim_amount = np.array(claim_amount)


claim_type = []

for i in range(N):
    if claim_occurred[i] == 1:
        if tracking_device[i] == 0 and crime_rate_area[i] == "high":
            claim_type.append(np.random.choice(["theft", "accident"], p=[0.6, 0.4]))
        elif flood_risk_area[i] == 1:
            claim_type.append(np.random.choice(["flood", "accident"], p=[0.5, 0.5]))
        else:
            claim_type.append("accident")
    else:
        claim_type.append("none")

df = pd.DataFrame({
    "age": ages,
    "driver_type": driver_type,
    "years_experience": years_experience,
    "income_band": income_band,
    "vehicle_age": vehicle_age,
    "vehicle_type": vehicle_type,
    "vehicle_value": vehicle_value,
    "airbags": airbags,
    "tracking_device": tracking_device,
    "region": region,
    "traffic_density": traffic_density,
    "policy_duration": policy_duration,
    "annual_mileage": annual_mileage,
    "speeding_score": speeding_score,
    "previous_claims": previous_claims,
    "fraud_flag": fraud_flag,
    "risk_score": risk_score,
    "base_premium": base_premium,
    "premium": premium,
    "claim_occurred": claim_occurred,
    "claim_amount": claim_amount,
    "claim_type": claim_type
})

df.to_csv("insurance_dataset.csv", index=False)

print("Dataset created successfully!")