import pandas as pd
import os

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_PATH = os.path.join(DATA_DIR, "raw_startup_data.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_startup_data.csv")

print("Reading data from:")
print(RAW_PATH)

# =============================
# Load data
# =============================
df = pd.read_csv(RAW_PATH)
print("Original dataset shape:", df.shape)

# =============================
# CLEAN COLUMN NAMES (IMPORTANT)
# =============================
# removes hidden spaces like "  latitude"
df.columns = df.columns.str.strip()

# =============================
# CREATE TARGET VARIABLE
# =============================
if "status" not in df.columns:
    raise ValueError("Column 'status' not found!")

# closed → failed (1), else → not failed (0)
df["failed"] = df["status"].astype(str).str.lower().apply(
    lambda x: 1 if x == "closed" else 0
)

print("Target variable 'failed' created.")

# =============================
# DROP NON-ML COLUMNS
# =============================
DROP_COLUMNS = [
    "Unnamed: 0",
    "name",
    "permalink",
    "homepage_url",
    "twitter_username",
    "state_code",
    "city",
    "status",
    "closed_at",
    "founded_at",
    "first_funding_at",
    "last_funding_at",
    "first_milestone_at",
    "last_milestone_at",
    "labels"  # safety drop if present
]

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
print("After dropping columns:", df.shape)

# =============================
# KEEP ONLY NUMERIC FEATURES
# =============================
df = df.select_dtypes(include=["number"])
print("After keeping numeric columns only:", df.shape)

# =============================
# FINAL SAFETY CLEAN
# =============================
# remove any Unnamed columns that sneak in
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# fill missing values
df = df.fillna(0)

# =============================
# SAVE
# =============================
df.to_csv(PROCESSED_PATH, index=False)

print("Processed dataset saved to:")
print(PROCESSED_PATH)
