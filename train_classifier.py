"""
Step 2: train the planet-vs-false-alarm classifier (run once).

Uses data/kepler.csv (made by fetch_data.py). If you already have your own
Kepler CSV from your old project, put it at data/kepler.csv instead, it just
needs the same column names (see FEATURES below).

Run:  python train_classifier.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_steff", "koi_slogg", "koi_srad",
]
LABEL = "koi_disposition"

df = pd.read_csv("data/kepler.csv")
df = df[df[LABEL].isin(["CONFIRMED", "FALSE POSITIVE"])]
df = df.dropna(subset=FEATURES)
print(f"Rows used: {len(df)}")

X = df[FEATURES]
y = (df[LABEL] == "CONFIRMED").astype(int)  # 1 = real planet

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

print(classification_report(
    y_test, model.predict(X_test),
    target_names=["False positive", "Confirmed planet"],
))

joblib.dump({"model": model, "features": FEATURES}, "model.joblib")
print("Saved model.joblib")
