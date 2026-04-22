# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")
df = pd.read_csv("loan_risk_prediction_dataset.csv")

print(f"Dataset shape: {df.shape}")

# Drop rows where target is missing; fill or drop other NaNs
df.dropna(subset=["LoanApproved"], inplace=True)

# Separate features and target
X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

# Fill missing values: mode for categoricals, median for numerics
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].fillna(X[col].mode()[0])

for col in X.select_dtypes(exclude="object").columns:
    X[col] = X[col].fillna(X[col].median())

# Identify column types
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

print(f"Numeric features   : {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# Build preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=42))
])

print("\nTraining model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# Save pipeline
joblib.dump(pipeline, "credit_risk_pipeline.pkl")
print("\nModel saved to credit_risk_pipeline.pkl — ready for prediction.")
