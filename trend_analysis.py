# -----------------------------------------------------------
# trend_analysis.py
# -----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_no_show_data.csv")

# Chart 1
plt.hist(df['Age'], bins=30)
plt.title("Age Distribution")
plt.show()

# Chart 2
df['No-show'].value_counts().plot(kind='bar')
plt.title("Show vs No-Show Count")
plt.show()

# Chart 3
df.groupby('SMS_received')['No-show'].mean().plot(kind='bar')
plt.title("SMS Impact on No-Show")
plt.show()

# Chart 4
plt.scatter(df['WaitingDays'], df['No-show'])
plt.title("Waiting Days vs No-Show")
plt.show()

# Chart 5
from sklearn.tree import DecisionTreeClassifier
features = ['Age','Scholarship','Hipertension','Diabetes','Alcoholism','SMS_received','WaitingDays']
X = df[features]
y = df['No-show']

model = DecisionTreeClassifier(max_depth=5)
model.fit(X, y)

plt.barh(features, model.feature_importances_)
plt.title("Feature Importance")
plt.show()
