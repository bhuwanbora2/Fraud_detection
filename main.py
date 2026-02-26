import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df["Class"].value_counts())

# -----------------------------
# 2️⃣ Split features and target
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount column
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# -----------------------------
# 3️⃣ Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4️⃣ Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# -----------------------------
# 5️⃣ Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6️⃣ Evaluate
# -----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7️⃣ SAVE MODEL (IMPORTANT)
# -----------------------------
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")