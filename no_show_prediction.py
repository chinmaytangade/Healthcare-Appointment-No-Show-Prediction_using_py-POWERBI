# -----------------------------------------------------------
# Healthcare Appointment No-Show Prediction Project
# Complete Python Code (Copy–Paste Ready)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------

df = pd.read_csv("KaggleV2-May-2016.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

# -----------------------------------------------------------
# 2. Data Cleaning
# -----------------------------------------------------------

# Drop unnecessary columns
df = df.drop(['PatientId', 'AppointmentID', 'Neighbourhood'], axis=1)

# Convert date columns
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Convert target column ("No"=0, "Yes"=1)
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

# Remove negative ages or impossible age values
df = df[df['Age'] >= 0]

# Create waiting days feature
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Handle negative values
df = df[df['WaitingDays'] >= 0]

print("\n--- Cleaned Data Columns ---")
print(df.columns)

# -----------------------------------------------------------
# 3. Feature Selection
# -----------------------------------------------------------

features = [
    'Age',
    'Scholarship',
    'Hipertension',
    'Diabetes',
    'Alcoholism',
    'SMS_received',
    'WaitingDays'
]

X = df[features]
y = df['No-show']

# -----------------------------------------------------------
# 4. Train/Test Split
# -----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 5. Decision Tree Model
# -----------------------------------------------------------

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------------------------------------
# 6. Evaluation
# -----------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------------------------------
# 7. Feature Importance Visualization
# -----------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.barh(features, model.feature_importances_)
plt.title("Feature Importance in No-Show Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 8. Save Cleaned Data for Power BI
# -----------------------------------------------------------

df.to_csv("cleaned_no_show_data.csv", index=False)
print("\nCleaned dataset saved as cleaned_no_show_data.csv")

# -----------------------------------------------------------
# END OF PROJECT
# -----------------------------------------------------------
