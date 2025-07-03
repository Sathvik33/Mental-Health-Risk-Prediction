import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score

# Base directory and dataset path
BASE_DIR = os.path.dirname(__file__)
Data_path = os.path.join(BASE_DIR, "Data", "mental_health_dataset.csv")

# Load dataset
Data = pd.read_csv(Data_path)

print(Data.head())
print(Data.info())
print(Data.describe())
print(Data.isnull().sum())
print(Data.shape)
print(Data.columns)
print(Data.dtypes)
print(Data['mental_health_risk'].value_counts())
print(Data['employment_status'].value_counts())
print(Data['work_environment'].value_counts())

cat_cols = ['gender', 'employment_status', 'work_environment', 'mental_health_history', 'seeks_treatment']
Data = pd.get_dummies(Data, columns=cat_cols, drop_first=True)

label_encoder = LabelEncoder()
Data['target'] = label_encoder.fit_transform(Data['mental_health_risk'])

X = Data.drop(['mental_health_risk', 'target'], axis=1)
y = Data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Prediction & Evaluation
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, average='weighted'))



importances = model.feature_importances_
features = X.columns

indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.title("Top 10 Important Features")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.tight_layout()
plt.show()

train_pred = model.predict(X_train)
print("Train Accuracy:", accuracy_score(y_train, train_pred))


corr = Data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "mental_health_model.pkl"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(model_dir, "model_features.pkl"))