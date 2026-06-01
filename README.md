# Customer Churn Prediction

> A production-ready machine learning system that predicts telecom customer churn using a tuned Random Forest classifier, served via a REST API and an interactive web interface.

**[Live Demo](https://moetezrouihem-dev.github.io/Customer_churn_prediction/)** · **[API Docs](https://customer-churn-prediction-xt7d.onrender.com/docs)**

---

## Overview

Predicts whether a telecom customer is likely to churn based on 19 behavioral and demographic features. It goes beyond a basic notebook — the model is deployed as a live REST API with a fully interactive frontend, making it usable by anyone without any technical knowledge.

---

## Demo

| Idle State | Prediction Result |
|---|---|
|Waiting for input | Risk level, probability arc, and churn verdict |

---

## Features

- **Model benchmarking** — Decision Tree, XGBoost, and Random Forest evaluated via 5-fold CV; Random Forest selected for best test set performance
- **Tuned Random Forest** via `RandomizedSearchCV` with 5-fold cross-validation
- **Class imbalance handling** using SMOTE (Synthetic Minority Oversampling)
- **One-Hot Encoding** for all categorical features with `drop_first=True`
- **REST API** built with FastAPI — fully documented at `/docs`
- **Risk classification** — High / Medium / Low based on churn probability
- **Interactive frontend** — particle background, animated probability arc, glassmorphism UI
- **Production deployment** — API on Render, frontend on GitHub Pages

---

## Model Performance

### Model Comparison (5-fold CV, scoring=accuracy)

| Model | CV Accuracy |
|---|---|
| Decision Tree | 0.80 |
| Random Forest | 0.84 |
| XGBoost | 0.84 |

Random Forest was selected over XGBoost despite identical CV accuracy — it achieved better F1 stability on the test set. Final tuning used `scoring=f1`.

### Tuned Random Forest — Test Set Results

| Metric | Default RF | Tuned RF |
|---|---|---|
| Accuracy | 0.79 | 0.79 |
| F1 Score (churn class) | 0.59 | 0.61 |
| ROC-AUC | 0.83 | 0.84 |

### Per-Class Breakdown (Tuned Model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Churn (0) | 0.86 | 0.85 | 0.86 |
| Churn (1) | 0.60 | 0.63 | 0.61 |

### Confusion Matrix
[[897  139]

[158  215]]

 **158** churners missed (false negatives — the costly error)
Tuning optimized for **F1 score** rather than accuracy,the correct choice for imbalanced classification where missing a churner is more costly than a false alarm.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Scikit-learn RandomForestClassifier |
| Tuning | RandomizedSearchCV |
| Imbalance | imbalanced-learn (SMOTE) |
| API | FastAPI + uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Deployment | Render (API) + GitHub Pages (frontend) |

---

## Project Structure

```
churn-api/
├── main.py                  # FastAPI application
├── churn_model.pkl          # Trained and serialized model
├── feature_names.json       # Column names for inference alignment
├── requirements.txt         # Python dependencies
├── index.html               # Frontend interface
└── Customer_Churn_Prediction.ipynb   # Full training notebook
```

---

## API Reference

**Base URL:** `https://customer-churn-prediction-xt7d.onrender.com`

### `GET /health`
Returns server status.

```json
{ "status": "healthy" }
```

### `POST /predict`
Predicts churn for a given customer profile.

**Request body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

**Response:**
```json
{
  "churn": true,
  "churn_probability": 0.7812,
  "risk_level": "High"
}
```

**Risk levels:**

| Level | Probability |
|---|---|
| High | ≥ 0.70 |
| Medium | 0.40 – 0.69 |
| Low | < 0.40 |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/moetezrouihem-dev/Customer_churn_prediction.git
cd Customer_churn_prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the API**
```bash
uvicorn main:app --reload
```

**4. Open the frontend**

Open `index.html` directly in your browser — it will call `http://127.0.0.1:8000/predict` by default.

**5. Explore the docs**

Navigate to `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## ML Pipeline

```
Raw CSV
  │
  ├── Drop customerID
  ├── Fix TotalCharges (whitespace → 0.0)
  ├── Encode target (Yes/No → 1/0)
  └── One-Hot Encode categorical features (drop_first=True)
        │
        ├── Train/Test Split (80/20, random_state=42)
        │
        └── SMOTE on training set only
              │
              ├── RandomizedSearchCV (30 iterations, cv=5, scoring=f1)
              │
              └── Best estimator → joblib.dump (compress=2)
```

---

## Dataset

**Telco Customer Churn** — IBM Sample Dataset via Kaggle

- 7,043 customers · 19 features · Binary target (Churn: Yes/No)
- Class distribution: 73% No Churn / 27% Churn → handled with SMOTE

---

## Author

**Moetez Rouihem** — CS Engineering Student at ENSI, Tunis

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/moetezrouihem-dev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/rouihem-moetez-10514b399/)

