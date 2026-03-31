import pandas as pd
import numpy as np

np.random.seed(42)

n_patients = 1000
n_appointments = 5000

# ================================
# 1. GENERATE PATIENT TABLE
# (one row per patient, stable attributes)
# ================================
patient_ids = np.arange(1000, 1000 + n_patients)

patients = pd.DataFrame({
    "patient_id": patient_ids,
    "age": np.random.randint(18, 80, n_patients),
    "gender": np.random.choice(["M", "F"], n_patients),
    "insurance_type": np.random.choice(["Private", "Medicaid", "Uninsured"], n_patients, p=[0.6, 0.3, 0.1]),
    "distance_to_clinic": np.round(np.random.exponential(scale=10, size=n_patients), 1),
    "has_transportation": np.random.choice([0, 1], n_patients, p=[0.2, 0.8]),
    "needs_interpreter": np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
    "employment_status": np.random.choice(["Employed", "Unemployed", "Part-time"], n_patients, p=[0.6, 0.2, 0.2]),
    "housing_stability": np.random.choice(["Stable", "Unstable"], n_patients, p=[0.85, 0.15]),
    "childcare_responsibility": np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
})

# ================================
# 2. GENERATE APPOINTMENT TABLE
# (one row per appointment, variable attributes)
# ================================
start_date = pd.Timestamp("2023-01-01")

appointments = pd.DataFrame({
    "appointment_id": range(1, n_appointments + 1),
    "patient_id": np.random.choice(patient_ids, n_appointments),
    "appointment_date": start_date + pd.to_timedelta(np.random.randint(0, 730, n_appointments), unit="D"),
    "appointment_type": np.random.choice(["Primary Care", "Specialty", "Mental Health", "Follow-up"], n_appointments),
    "day_of_week": np.random.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], n_appointments),
    "time_of_day": np.random.choice(["Morning", "Afternoon", "Evening"], n_appointments),
    "lead_time_days": np.random.randint(0, 60, n_appointments),
    "prior_no_show_rate": np.round(np.random.beta(2, 5, n_appointments), 2),
    "prior_visits": np.random.randint(1, 20, n_appointments),
    "last_appointment_no_show": np.random.choice([0, 1], n_appointments, p=[0.7, 0.3]),
    "reminder_sent": np.random.choice([0, 1], n_appointments, p=[0.2, 0.8]),
    "confirmation_received": np.random.choice([0, 1], n_appointments, p=[0.4, 0.6]),
})

# ================================
# 3. MERGE PATIENT ATTRIBUTES ONTO APPOINTMENTS
# ================================
df = appointments.merge(patients, on="patient_id", how="left")

# ================================
# 4. GENERATE NO-SHOW PROBABILITY
# ================================
prob = (
    0.15
    + 0.25 * df["prior_no_show_rate"]
    + 0.10 * (df["lead_time_days"] > 30)
    + 0.10 * (df["insurance_type"] != "Private")
    + 0.10 * (df["has_transportation"] == 0)
    + 0.08 * (df["confirmation_received"] == 0)
    + 0.05 * (df["reminder_sent"] == 0)
    + 0.05 * (df["distance_to_clinic"] > 15)
    + 0.05 * (df["appointment_type"] == "Mental Health")
    + 0.03 * (df["day_of_week"].isin(["Mon", "Fri"]))
    + 0.08 * (df["employment_status"] != "Employed")
    + 0.07 * (df["housing_stability"] == "Unstable")
    + 0.05 * (df["childcare_responsibility"] == 1)
)

prob = np.clip(prob, 0, 0.95)
df["no_show"] = np.random.binomial(1, prob)

# ================================
# 5. SAVE
# ================================
print(df.head())
print(f"\nShape: {df.shape}")
df.to_csv("synthetic_appointment_data.csv", index=False)