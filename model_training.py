# -----------------------------------------------------------
# model_training.py
# -----------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("cleaned_no_show_data.csv")

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

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
