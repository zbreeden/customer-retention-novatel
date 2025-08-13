# churn_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('data/placeholder.csv')

# Basic data cleaning
df.dropna(inplace=True)  # Remove rows with missing values

# Convert 'churn' column to binary
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# Define features and target
X = df[['tenure', 'monthly_charges']]
y = df['churn']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
