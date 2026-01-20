import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_startup_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading processed data from:")
print(DATA_PATH)

# =============================
# Load data
# =============================
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# =============================
# Split features and target
# =============================
if "failed" not in df.columns:
    raise ValueError("Target column 'failed' not found in processed data!")

X = df.drop(columns=["failed"])
y = df["failed"]

print("Number of features:", X.shape[1])

# =============================
# Train-test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =============================
# Train model
# =============================
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

model.fit(X_train, y_train)

# =============================
# Evaluate model
# =============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# =============================
# Save model
# =============================
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
