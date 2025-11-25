# -----------------------------------------------------------
# main.py
# RUNS THE COMPLETE HEALTHCARE NO-SHOW PROJECT
# -----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n=======================================")
print(" STEP 1: CLEANING DATA ")
print("=======================================\n")

# -----------------------------------------------------------
# 1. DATA CLEANING
# -----------------------------------------------------------

df = pd.read_csv("KaggleV2-May-2016.csv")

df = df.drop(['PatientId', 'AppointmentID', 'Neighbourhood'], axis=1)

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

df = df[df['Age'] >= 0]

df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df[df['WaitingDays'] >= 0]

df.to_csv("cleaned_no_show_data.csv", index=False)

print("✔ Data cleaned and saved as cleaned_no_show_data.csv\n")


print("\n=======================================")
print(" STEP 2: TRAINING MODEL ")
print("=======================================\n")

# -----------------------------------------------------------
# 2. MODEL TRAINING
# -----------------------------------------------------------

features = [
    'Age', 'Scholarship', 'Hipertension', 
    'Diabetes', 'Alcoholism', 
    'SMS_received', 'WaitingDays'
]

X = df[features]
y = df['No-show']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



print("\n=======================================")
print(" STEP 3: TREND ANALYSIS + 5 CHARTS ")
print("=======================================\n")

# -----------------------------------------------------------
# 3. CHARTS
# -----------------------------------------------------------

# Chart 1 - Age Distribution
plt.hist(df['Age'], bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Chart 2 - No-show Count
df['No-show'].value_counts().plot(kind='bar')
plt.title("Show vs No-Show Count")
plt.xlabel("0 = Show, 1 = No-Show")
plt.ylabel("Count")
plt.show()

# Chart 3 - SMS vs No-show Rate
df.groupby('SMS_received')['No-show'].mean().plot(kind='bar')
plt.title("SMS Impact on No-Show Rate")
plt.xlabel("SMS Received (0/1)")
plt.ylabel("No-Show Rate")
plt.show()

# Chart 4 - Waiting Days vs No-show
plt.scatter(df['WaitingDays'], df['No-show'])
plt.title("Waiting Days vs No-Show")
plt.xlabel("Waiting Days")
plt.ylabel("No-Show (0/1)")
plt.show()

# Chart 5 - Feature Importance
plt.barh(features, model.feature_importances_)
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

print("\n✔ ALL TASKS COMPLETED SUCCESSFULLY!")
