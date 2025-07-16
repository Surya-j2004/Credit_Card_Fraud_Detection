import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv('creditcard.csv')

# Balance the dataset (undersample)
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=492, random_state=42)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

FEATURE_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

X = balanced_data[FEATURE_COLUMNS]
y = balanced_data['Class']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Print metrics
print("Training complete.")
print(classification_report(y_test, model.predict(X_test_scaled)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test_scaled)))
