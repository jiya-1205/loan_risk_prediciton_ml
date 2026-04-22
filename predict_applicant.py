# predict_applicant.py

import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load("credit_risk_pipeline.pkl")

print("=" * 45)
print("       Loan Risk Prediction System")
print("=" * 45)
print("Enter applicant details:\n")

# Collect inputs matching the exact dataset columns:
# Age, Income, LoanAmount, CreditScore, YearsExperience,
# Gender, Education, City, EmploymentType
data = {
    "Age":             [int(input("Age: "))],
    "Income":          [float(input("Annual Income: "))],
    "LoanAmount":      [float(input("Loan Amount: "))],
    "CreditScore":     [int(input("Credit Score (300-850): "))],
    "YearsExperience": [int(input("Years of Work Experience: "))],
    "Gender":          [input("Gender (Male/Female): ").strip()],
    "Education":       [input("Education (High School/Bachelor's/Master's/PhD): ").strip()],
    "City":            [input("City (e.g. New York/Houston/San Francisco/Chicago/Phoenix): ").strip()],
    "EmploymentType":  [input("Employment Type (Full-Time/Part-Time/Self-Employed/Unemployed): ").strip()],
}

sample = pd.DataFrame(data)

prediction = pipeline.predict(sample)
probability = pipeline.predict_proba(sample)

approved_prob = probability[0][1] * 100
rejected_prob = probability[0][0] * 100

print("\n" + "=" * 45)
if prediction[0] == 1:
    print("✅  LOAN APPROVED  (Low Risk)")
else:
    print("❌  LOAN REJECTED  (High Risk)")

print(f"\n   Approval Probability : {approved_prob:.1f}%")
print(f"   Rejection Probability: {rejected_prob:.1f}%")
print("=" * 45)
