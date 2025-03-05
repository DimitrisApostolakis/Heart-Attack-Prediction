import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data.csv')

# Fixing the name of the column 'num'
df.columns = df.columns.str.strip()
# Removing unfilled & unnecessary columns
df.drop(columns=['slope', 'ca', 'thal'], inplace=True)

# Cleaning the table from NaN
# print(df.info())
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.fillna(df.mean(), inplace=True)

# print("How many ? are in the table: ", df.isin(["?"]).sum().sum())

X = df.drop(columns=['num']) # Retrieving the features
y = df['num'] # 0: Low risk, 1: High risk

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.joblib')

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuray of the model: {acc:.2f}")

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "heart_attack_model.joblib")

