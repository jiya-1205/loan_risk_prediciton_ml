# credit_risk_algozee.py
# Using Kaggle Algozee Credit Risk & Loan Default Analysis Dataset

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve

# 1) Load dataset
print("Loading dataset...")
df = pd.read_csv("loan_risk_prediction_dataset.csv")
print(df.head())

# 2) Clean missing values
print("Cleaning missing values…")
df.fillna(df.mode().iloc[0], inplace=True)

# 3) Separate target and features
target_col = "LoanApproved"
y = df[target_col]
X = df.drop(columns=[target_col])

# 4) Encode categorical features
print("Encoding categorical columns…")
le = LabelEncoder()
for col in X.select_dtypes(include=['object', 'string']).columns:
    X[col] = le.fit_transform(X[col])

# 5) Scale numeric features
print("Scaling features…")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6) Train / test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("Training models…")

# 7) Train 3 algorithms
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 8) Evaluation
def evaluate(name, ytest, pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(ytest, pred))
    print("Confusion Matrix:\n", confusion_matrix(ytest, pred))
    print(classification_report(ytest, pred))

evaluate("Naive Bayes", y_test, nb_pred)
evaluate("KNN", y_test, knn_pred)
evaluate("Decision Tree (Balanced)", y_test, dt_pred)

# 9) ROC curve
print("\nSaving ROC curve…")
models = [nb, knn, dt]
names = ["Naive Bayes", "KNN", "Decision Tree"]

for model, name in zip(models, names):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_curve.png")

# 10) Final comparison
results = pd.DataFrame({
    "Model": ["Naive Bayes", "KNN", "Decision Tree (Balanced)"],
    "Accuracy": [
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, dt_pred)
    ]
})

print("\nFinal Comparison:\n", results)