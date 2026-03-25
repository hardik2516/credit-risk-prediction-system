# 💳 Credit Risk Prediction System

🔗 **Live Demo:** https://huggingface.co/spaces/Hardik-25/credit-risk-prediction

---

## 📌 Overview

This project is an end-to-end Machine Learning system designed to predict the probability of loan default using customer financial and behavioral data.

It simulates a real-world credit risk evaluation system used by financial institutions.

---

## 🎯 Problem Statement

Financial institutions need to assess whether a customer is likely to default on a loan.

The challenge:

* Incomplete credit history for some users
* Complex financial behavior patterns
* Risk of financial loss due to incorrect approvals

---

## 💡 Solution

This system uses **two ML models**:

### ✅ Model A (Primary Model)

* Uses full credit history (EXT_SOURCE features)
* More accurate predictions
* Used for regular applicants

### ⚠️ Model B (Fallback Model)

* Works without credit history
* Used for new borrowers
* Acts as a screening model

---

## ⚙️ Features

* 📊 Real-time default probability prediction
* 🧠 Dual-model architecture
* 📉 Risk categorization (Low / Moderate / High)
* 📌 Feature engineering for financial ratios
* 🔍 Top 3 reasons for risk prediction (explainability)
* 🎯 Custom decision thresholds
* 💻 Interactive Streamlit UI

---

## 🏗️ Project Structure

```
credit-risk-prediction-system/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── models/
│   ├── best_model.pkl
│   ├── model_no_ext.pkl
│   ├── threshold.pkl
│   ├── threshold_no_ext.pkl
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_feature_engineering_modeling.ipynb
│
├── data/
│   ├── raw/
│   │   └── application_train.csv
│   ├── processed/
│       └── cleaned_data.csv
```

---

## 📊 Dataset

Dataset used: **Home Credit Default Risk**

🔗 https://www.kaggle.com/competitions/home-credit-default-risk

> Note: Dataset not included due to size constraints.

---

## 🧠 Feature Engineering

Key engineered features:

* EMI to Income Ratio
* Loan to Income Ratio
* Total Payment Burden
* Income per Family Member
* Employment Stability
* External Credit Score Aggregations

---

## 🤖 Model Details

* Algorithm: Gradient Boosting / XGBoost
* Evaluation Metric: ROC-AUC
* Threshold tuning for better decision making
* Separate pipelines for both models

---

## 📈 Risk Interpretation

| Probability      | Risk Level    |
| ---------------- | ------------- |
| < 0.30           | Low Risk      |
| 0.30 – Threshold | Moderate Risk |
| > Threshold      | High Risk     |

---

## 🚀 Deployment

* Platform: Hugging Face Spaces
* Framework: Streamlit
* Supports real-time predictions

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* SHAP

---

## ⚠️ Limitations

* Model B has lower accuracy due to missing credit history
* Dataset imbalance may affect predictions
* External economic factors not included

---

## 🔮 Future Improvements

* API deployment (FastAPI)
* Enhanced explainability
* Model monitoring
* Cloud database integration

---

## 👨‍💻 Author

**Hardik Gautam**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
