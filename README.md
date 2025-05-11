# AI-driven-traffic-accident-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, RocCurveDisplay
import numpy as np

# 1. Load data
df = pd.read_csv("traffic_accidents.csv")

# 2. Select a compact set of useful columns
cols = [
    "crash_hour",
    "weather_condition",
    "lighting_condition",
    "trafficway_type",
    "roadway_surface_cond",
    "most_severe_injury",
]
df = df[cols].dropna()

# 3. Target encoding
label = LabelEncoder()
df["target"] = label.fit_transform(df["most_severe_injury"])
df = df.drop(columns=["most_severe_injury"])

# 4. One-hot-encode categoricals
df = pd.get_dummies(df, drop_first=True)

# 5. Scale crash_hour
scaler = MinMaxScaler()
df[["crash_hour"]] = scaler.fit_transform(df[["crash_hour"]])

# 6. Train / Test split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Model training (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 8. Predictions & evaluation
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# 9. Visualisations
## Confusion matrix
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted"), ax.set_ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

## ROC (if binary)
if len(label.classes_) == 2:
    RocCurveDisplay.from_estimator(rf, X_test, y_test)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()
