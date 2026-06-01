# ChurnRadar — Customer Churn Prediction

> A production-ready machine learning system that predicts telecom customer churn using a tuned Random Forest classifier, served via a REST API and an interactive web interface.

**[Live Demo](https://moetezrouihem-dev.github.io/Customer_churn_prediction/)** · **[API Docs](https://customer-churn-prediction-xt7d.onrender.com/docs)**

---

## Overview

ChurnRadar predicts whether a telecom customer is likely to churn based on 19 behavioral and demographic features. It goes beyond a basic notebook — the model is deployed as a live REST API with a fully interactive frontend, making it usable by anyone without any technical knowledge.

---

## Demo

| Idle State | Prediction Result |
|---|---|
| Radar scanner animation, waiting for input | Risk level, probability arc, and churn verdict |

---

## Features

- **Tuned Random Forest** via `RandomizedSearchCV` with 5-fold cross-validation
- **Class imbalance handling** using SMOTE (Synthetic Minority Oversampling)
- **One-Hot Encoding** for all categorical features with `drop_first=True`
- **REST API** built with FastAPI — fully documented at `/docs`
- **Risk classification** — High / Medium / Low based on churn probability
- **Interactive frontend** — particle background, animated probability arc, glassmorphism UI
- **Production deployment** — API on Render, frontend on GitHub Pages

---

## Model Performance

| Metric | Default RF | Tuned RF |
|---|---|---|
| Accuracy | 0.79 | 0.79 |
| F1 Score | 0.59 | 0.62 |
| ROC-AUC | 0.83 | 0.84 |

Tuning optimized for **F1 score** rather than accuracy — the correct choice for imbalanced classification where missing a churner is more costly than a false alarm.

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
| Serialization | joblib (compress=2) |

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
git clone https://github.com/YOURUSERNAME/churn-api.git
cd churn-api
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

## Key Design Decisions

**Why F1 over Accuracy?**
The dataset is imbalanced (73/27). A model predicting "No Churn" for everyone achieves 73% accuracy while being completely useless. F1 penalizes this behavior and rewards catching actual churners.

**Why One-Hot over Label Encoding?**
Categorical features like `Contract` (Month-to-month / One year / Two year) have no meaningful order. Label Encoding would imply `Two year = 2 × Month-to-month`, which is mathematically incorrect for tree-based models and especially wrong for linear ones.

**Why SMOTE over class_weight?**
SMOTE generates synthetic minority samples, giving the model more diverse examples of churners to learn from. `class_weight='balanced'` is simpler but only reweights — it doesn't enrich the feature space.

---

## Author

**Moetez Rouihem** — CS Engineering Student at ENSI, Tunis

[![GitHub](https://img.shields.io/badge/GitHub-moetezrouihem-dev-181717?style=flat&logo=github)](https://github.com/moetezrouihem-dev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/rouihem-moetez-10514b399/)

---

*Built as part of a production ML portfolio — model training, API development, and full deployment from scratch.*
