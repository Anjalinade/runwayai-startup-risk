import pandas as pd
import os
import joblib

# ==============================
# Resolve paths safely
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed_startup_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")

print("Loading model from:")
print(MODEL_PATH)

print("Loading data from:")
print(DATA_PATH)

# ==============================
# Load model and data
# ==============================
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop("failed", axis=1)

# ==============================
# Get feature importance
# ==============================
coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "weight": coefficients
})

# Sort by absolute importance
feature_importance["abs_weight"] = feature_importance["weight"].abs()
feature_importance = feature_importance.sort_values(
    by="abs_weight", ascending=False
)

# ==============================
# Print explanations (NO emojis)
# ==============================
print("\nTop 10 features INCREASING failure risk:")
print(feature_importance.head(10)[["feature", "weight"]])

print("\nTop 10 features REDUCING failure risk:")
print(feature_importance.tail(10)[["feature", "weight"]])
