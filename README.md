# Mental-Health-Risk-Prediction

# 🧠 BrainWave: Mental Health Risk Classifier

BrainWave is a machine learning–powered web app that predicts the mental health risk level (Low, Medium, High) based on user lifestyle and psychological inputs.

Built with:
- 🔍 XGBoost Classifier
- 📊 Streamlit for UI
- 🧠 One-hot encoding and feature scaling
- ✅ Class imbalance handling

---

## 🚀 Features

- Predicts mental health risk using real-world psychological and lifestyle indicators.
- User-friendly Streamlit interface.
- Balanced classification using XGBoost and sample weighting.
- Clean UI with dynamic risk display.

---

## 📁 Project Structure

BrainWave/
├── Data/
│ └── mental_health_dataset.csv
├── model/
│ ├── mental_health_model.pkl
│ ├── label_encoder.pkl
│ └── model_features.pkl
├── Mental Health.py ← Model training & preprocessing
├── M-app.py ← Streamlit web app
├── README.md
