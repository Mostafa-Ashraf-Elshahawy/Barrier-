# 🛡️ BARRIER // Customer Churn Intelligence & Retention Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Mostafa-Ashraf-Elshahawy%2FBarrier-&branch=main&mainModule=app.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-interactive%20viz-blueviolet.svg)](https://plotly.com/python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade Customer Churn Prediction and Proactive Retention Intelligence platform developed as part of the **NTI Graduation Project (Barrier-)**. Powered by tuned machine learning ensemble pipelines, interactive behavioral analytics, and a real-time **What-If Retention Simulator**.

---

## 🚀 Key Features

### 1. 🔮 Real-Time Customer Risk Scoring & Gauge
- Score any customer profile instantly across Contract duration, Tenure, Payment Method, Monthly Charges, and services.
- Dynamic gauge indicator visualizing churn probability with three distinct risk tiers (🟢 **Low Risk**, 🟡 **Moderate / At-Risk**, 🔴 **High Risk**).
- Automated risk-driver attribution explaining which specific factors drive up churn risk.
- Prescriptive retention playbooks recommending tailored intervention actions.

### 2. 🧪 Interactive "What-If" Retention Simulator
- Test hypothetical retention strategies in real-time.
- Adjust contract commitment, payment method, or apply promotional discounts to immediately observe the reduction in churn probability.

### 3. 📁 Batch Customer Scoring & Portfolio Risk
- Upload a custom CSV or evaluate sample customer portfolios directly from the database.
- Calculates portfolio-wide KPIs: Total Scored, Predicted Churners, High-Risk Cohort Count, and **Monthly Revenue at Risk ($)**.
- Filter records by risk category and export fully scored customer lists as downloadable CSV files.

### 4. 📊 Exploratory Data Analysis (EDA) & Insights
- Interactive Plotly visualizations for churn rates across Contract types, Payment Methods, Tenure Cohorts, and Billing distributions.
- Business intelligence metrics tracking overall churn rate and annualized revenue loss.

### 5. 🤖 Comprehensive Model Benchmarks & Explainability
- Benchmark table across **8 supervised learning algorithms** evaluated via 5-Fold Stratified Cross-Validation (`GridSearchCV`).
- Interactive Confusion Matrix and Global Feature Importance rankings.
- Dynamic model selector allowing switching between **Random Forest** (top performer) and **XGBoost**.

---

## 📊 Model Evaluation & Benchmarks

The models were evaluated on an independent 20% holdout test set (1,409 customers) using 5-Fold Stratified Cross-Validation:

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Optimal Hyperparameters |
|---|---|---|---|---|---|---|---|
| **Random Forest** 🏆 | **0.8399** | **0.8446** | 76.22% | 53.69% | **75.94%** | **0.6290** | `max_depth: 8, n_estimators: 200, class_weight: 'balanced'` |
| **XGBoost** | **0.8405** | **0.8433** | **79.77%** | **66.54%** | 47.86% | 0.5568 | `learning_rate: 0.05, max_depth: 3, n_estimators: 200` |
| **Gradient Boosting** | 0.8390 | 0.8399 | 79.56% | 65.81% | 47.86% | 0.5542 | `learning_rate: 0.1, max_depth: 3, n_estimators: 100` |
| **Logistic Regression** | 0.8402 | 0.8373 | 73.95% | 50.59% | 79.68% | 0.6189 | `C: 0.1, class_weight: 'balanced'` |
| **SVM** | 0.8396 | 0.8368 | 72.89% | 49.37% | 84.22% | 0.6225 | `C: 0.1, kernel: 'linear'` |
| **AdaBoost** | 0.8397 | 0.8358 | 79.06% | 64.26% | 47.59% | 0.5469 | `learning_rate: 1.0, n_estimators: 200` |
| **Naive Bayes** | 0.8179 | 0.8162 | 74.66% | 51.63% | 72.19% | 0.6020 | Default GaussianNB |
| **KNN** | 0.8075 | 0.8121 | 78.21% | 60.84% | 50.27% | 0.5505 | `n_neighbors: 9` |

---

## 🛠️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Mostafa-Ashraf-Elshahawy/Barrier-.git
cd Barrier-
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Fork or push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New App"** and select:
   - **Repository:** `Mostafa-Ashraf-Elshahawy/Barrier-`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml                  # Custom styling and server configuration
├── artifacts/
│   └── model_comparison.csv         # Benchmark evaluation metrics across 8 algorithms
├── data/
│   ├── churn_data.csv               # Raw customer dataset (7,042 records)
│   └── processed_churn_data.csv     # Preprocessed dataset with engineered features
├── models/
│   └── churn_model_package.pkl      # Pre-trained Random Forest model and scaler
├── notebooks/
│   ├── preproccing.ipynb            # Data cleaning, EDA & feature transformation
│   └── training models.ipynb        # Model training, 5-Fold GridSearchCV & evaluation
├── app.py                           # Main Streamlit web application
├── requirements.txt                 # Application dependencies
└── README.md                        # Documentation & project guide
```

---

## 👥 Contributors & Acknowledgements
Developed with ❤️ for the **NTI Graduation Project (Barrier-)**.
- **Repository:** [https://github.com/Mostafa-Ashraf-Elshahawy/Barrier-](https://github.com/Mostafa-Ashraf-Elshahawy/Barrier-)
