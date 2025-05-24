import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("final_diabetes_data.csv")

# Define features & target
features = ["Age", "BMI", "HbA1c", "Hypertension", "FamilyHistoryDiabetes", "DietQuality"]
target = "Diagnosis"

# Encode categorical variables
label_encoders = {}
for col in ["FamilyHistoryDiabetes", "DietQuality", "Diagnosis"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & encoders
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save Accuracy & Confusion Matrix Plot
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Model Accuracy: {accuracy:.2%}")
plt.savefig("static/confusion_matrix.png")

# Save feature importance graph
feature_importance = pd.Series(model.feature_importances_, index=features)
plt.figure(figsize=(6, 4))
feature_importance.sort_values().plot(kind="barh", color="teal")
plt.xlabel("Importance Score")
plt.title("Feature Importance")
plt.savefig("static/feature_importance.png")

print("Model trained and saved successfully!")
