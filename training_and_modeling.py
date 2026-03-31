import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("synthetic_appointment_data.csv")

df["appointment_date"] = pd.to_datetime(df["appointment_date"])
df["distance_to_clinic"] = df["distance_to_clinic"].astype(float)
df["employment_status"] = df["employment_status"].astype(str)
df["housing_stability"] = df["housing_stability"].astype(str)
df["childcare_responsibility"] = df["childcare_responsibility"].astype(int)

# ================================
# ENGINEER HISTORY-BASED FEATURES
# ================================
# For each appointment, compute the patient's history UP TO that appointment
# (excluding the current one, to avoid data leakage)

history_features = []

for idx, row in df.iterrows():
    past = df[
        (df["patient_id"] == row["patient_id"]) &
        (df["appointment_date"] < row["appointment_date"])
    ]
    
    if past.empty:
        actual_no_show_rate = row["prior_no_show_rate"]  # fall back to synthetic value
        total_visits = row["prior_visits"]
        last_was_no_show = row["last_appointment_no_show"]
    else:
        actual_no_show_rate = past["no_show"].mean()
        total_visits = len(past)
        last_was_no_show = int(past.sort_values("appointment_date").iloc[-1]["no_show"])
    
    history_features.append({
        "idx": idx,
        "actual_no_show_rate": actual_no_show_rate,
        "total_visits": total_visits,
        "last_was_no_show": last_was_no_show
    })

history_df = pd.DataFrame(history_features).set_index("idx")

# Replace synthetic history columns with real calculated ones
df["prior_no_show_rate"] = history_df["actual_no_show_rate"]
df["prior_visits"] = history_df["total_visits"]
df["last_appointment_no_show"] = history_df["last_was_no_show"]

# ================================
# PREPARE DATA FOR MODEL
# ================================
df_encoded = pd.get_dummies(df.drop(columns=["appointment_date"]), drop_first=True)

X = df_encoded.drop(["no_show", "appointment_id"], axis=1)
y = df_encoded["no_show"]

# ================================
# TRAIN MODEL
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Model training complete.")

# ================================
# PREDICTION FUNCTION
# ================================
def predict_no_show_risk(patient_id):
    patient_rows = df[df["patient_id"] == patient_id]

    if patient_rows.empty:
        return "Patient ID not found."

    history = patient_rows.sort_values("appointment_date")

    # Compute real history features from all their appointments
    actual_no_show_rate = history["no_show"].mean()
    total_visits = len(history)
    last_was_no_show = int(history.iloc[-1]["no_show"])

    # Start from most recent appointment for demographic/appointment features
    latest = history.iloc[-1].copy()
    latest["prior_no_show_rate"] = actual_no_show_rate
    latest["prior_visits"] = total_visits
    latest["last_appointment_no_show"] = last_was_no_show

    # Encode for model
    patient_df = pd.DataFrame([latest]).drop(columns=["appointment_date"])
    patient_encoded = pd.get_dummies(patient_df)
    patient_encoded = patient_encoded.reindex(columns=X.columns, fill_value=0)

    risk = float(model.predict_proba(patient_encoded)[0][1])

    # Risk level
    if risk < 0.3:
        level = "Low Risk"
    elif risk < 0.6:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    # -------------------------------
    # Identify contributing factors
    # -------------------------------
    reasons = []
    recommendations = []

    distance      = float(latest["distance_to_clinic"])
    prior_no_show = float(actual_no_show_rate)
    lead_time     = float(latest["lead_time_days"])
    has_transport = int(latest["has_transportation"])
    confirmed     = int(latest["confirmation_received"])
    childcare     = int(latest["childcare_responsibility"])
    employment    = str(latest["employment_status"])
    housing       = str(latest["housing_stability"])
    interpreter   = int(latest["needs_interpreter"])
    insurance     = str(latest["insurance_type"])

    if prior_no_show > 0.5:
        reasons.append("History of missed appointments")
        recommendations.append("Require confirmation call or double reminders")

    if lead_time > 30:
        reasons.append("Long wait time before appointment")
        recommendations.append("Schedule closer to appointment date if possible")

    if has_transport == 0:
        reasons.append("Transportation barrier")
        recommendations.append("Offer telehealth or transportation assistance")

    if distance > 15:
        reasons.append("Lives far from clinic")
        recommendations.append("Offer telehealth or closer clinic location")

    if confirmed == 0:
        reasons.append("No appointment confirmation")
        recommendations.append("Send confirmation request or follow-up call")

    if insurance != "Private":
        reasons.append("Insurance-related access challenges")
        recommendations.append("Provide additional outreach or flexible scheduling")

    if employment != "Employed":
        reasons.append("Work schedule instability")
        recommendations.append("Offer flexible or after-hours appointments")

    if housing == "Unstable":
        reasons.append("Housing instability")
        recommendations.append("Provide flexible scheduling and outreach support")

    if childcare == 1:
        reasons.append("Childcare responsibilities")
        recommendations.append("Offer flexible timing or telehealth options")

    if interpreter == 1:
        reasons.append("Language barrier")
        recommendations.append("Ensure interpreter services are scheduled")

    recommendations = list(set(recommendations))

    return {
        "patient_id": patient_id,
        "appointments_on_record": total_visits,
        "historical_no_show_rate": round(actual_no_show_rate, 3),
        "no_show_probability": round(risk, 3),
        "risk_level": level,
        "key_risk_factors": reasons,
        "recommended_actions": recommendations
    }

result = predict_no_show_risk(1020)
print(result)