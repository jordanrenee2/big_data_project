# Predicting Patient No-Shows: A Big Data Approach to Healthcare Scheduling
 
A machine learning prototype for predicting patient appointment no-shows and generating individualized, actionable intervention recommendations. Built using Python and logistic regression on a synthetic dataset simulating realistic healthcare scheduling patterns.
 
---
 
## Overview
 
Patient no-shows cost the U.S. healthcare system an estimated $150 billion annually and disrupt care continuity for vulnerable populations. Standard responses — blanket reminder calls and overbooking — treat no-shows as a fixed rate to compensate for rather than a pattern worth understanding.
 
This tool takes a different approach: rather than applying the same intervention to everyone, it assigns each patient an individualized risk score before their appointment and recommends targeted outreach based on the specific barriers driving that patient's risk.
 
---
 
## Features
 
- Synthetic dataset generation simulating 1,000 patients and 5,000 appointments
- Longitudinal feature engineering that computes behavioral history per appointment without data leakage
- Logistic regression model trained on appointment-level and patient-level features
- Risk classification: **Low** (<30%), **Moderate** (30–60%), and **High** (>60%)
- Per-patient intervention recommendations mapped to specific identified risk factors
- Interactive command-line interface for querying individual patient risk
---
 
## Project Structure
 
```
├── data_generation.py          # Generates synthetic patient and appointment data
├── training_and_modeling.py    # Feature engineering, model training, and prediction function
├── main.py                     # Interactive CLI for querying patient risk scores
├── synthetic_appointment_data.csv  # Generated dataset (created by data_generation.py)
└── README.md
```
 
---
 
## Getting Started
 
### Prerequisites
 
```bash
pip install pandas numpy scikit-learn
```
 
### Usage
 
**Step 1: Generate the synthetic dataset**
```bash
python data_generation.py
```
This creates `synthetic_appointment_data.csv` with 5,000 appointment records linked to 1,000 simulated patients.
 
**Step 2: Run the prediction tool**
```bash
python main.py
```
You will be prompted to enter a patient ID (ranging from 1000–1999). The tool will return:
- No-show probability
- Risk level (Low / Moderate / High)
- Key contributing risk factors
- Recommended interventions
**Example output:**
```
--- No-Show Risk Prediction Tool ---
Enter patient ID (or 'q' to quit): 1042
 
Patient ID: 1042
No-show probability: 71.4%
Risk level: High Risk
Key risk factors:
  - History of missed appointments
  - Transportation barrier
  - Housing instability
Recommended actions:
  - Offer telehealth or transportation assistance
  - Provide flexible scheduling and outreach support
  - Require confirmation call or double reminders
```
 
---
 
## Features Used in the Model
 
| Category | Variables |
|---|---|
| Appointment-level | Lead time, day of week, time of day, appointment type |
| Behavioral history | Prior no-show rate, total visit count, last appointment outcome |
| Logistical | Distance to clinic, transportation access |
| Socioeconomic | Insurance type, employment status, housing stability, childcare responsibility |
| Operational | Reminder sent, confirmation received, interpreter needed |
 
---
 
## Intervention Logic
 
Each risk factor identified for a patient maps to a specific recommended action:
 
| Risk Factor | Recommended Intervention |
|---|---|
| No-show rate > 50% | Confirmation call or double reminders |
| Lead time > 30 days | Schedule closer to appointment date |
| No transportation | Telehealth or transportation assistance |
| Distance > 15 miles | Telehealth or closer clinic location |
| No confirmation received | Follow-up outreach call |
| Non-private insurance | Additional outreach or flexible scheduling |
| Non-employed | Flexible or after-hours appointments |
| Unstable housing | Flexible scheduling and outreach support |
| Childcare responsibility | Flexible timing or telehealth options |
| Interpreter needed | Pre-scheduled interpreter services |
 
---
 
## Ethical Considerations
 
This tool uses socioeconomic variables (insurance type, housing stability, employment status) as predictors. These are included to identify patients who may need additional support — **not** to deprioritize scheduling or ration access to care. Any real deployment should include:
 
- Clear institutional guidelines governing how predictions can and cannot be used
- Transparency with both providers and patients
- Ongoing auditing for disparate impact across demographic groups
- HIPAA-compliant data handling and storage
---
 
## Limitations
 
- Built on synthetic data; real-world performance would require validation against actual EHR data with proper IRB approval and HIPAA compliance infrastructure
- Logistic regression is interpretable but may underperform more complex models (e.g., random forests, gradient boosting) on larger datasets
- The model does not currently account for clinic-level or provider-level variation
---
 
## Future Directions
 
- Integration with EHR systems to generate risk scores at the point of scheduling
- Exploration of more sophisticated algorithms where higher accuracy justifies reduced interpretability
- Selective overbooking logic based on predicted risk distribution across time slots
- Automated outreach workflow triggers
---
 
## References
 
Chen, J., et al. (2021). Application of machine learning to predict patient no-shows in an academic pediatric ophthalmology clinic. *AMIA Symposium*, 293–302.
 
Deina, C., et al. (2024). Decision analysis framework for predicting no-shows to appointments using machine learning algorithms. *BMC Health Services Research*, 24(37).
 
Liu, D., et al. (2022). Machine learning approaches to predicting no-shows in pediatric medical appointment. *Digital Medicine*, 50.
