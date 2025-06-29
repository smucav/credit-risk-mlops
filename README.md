# Credit Risk Probability Model for Alternative Data

## 📌 Overview
An end-to-end credit risk scoring system for buy-now-pay-later (BNPL) services. Predicts customer default probability using alternative data (transactional/behavioral) and suggests optimal loan terms. Built with MLOps practices for reproducibility.

**Key Features**:
- 🎯 RFM-based credit risk proxy
- 🤖 Automated model training/prediction
- 🚀 FastAPI deployment
- 🔄 CI/CD integrated
- 📊 Comprehensive EDA notebooks


## 🏗️ Project Structure
```
credit-risk-mlops/
├── .github/workflows/ci.yml       # CI/CD pipeline
├── data/                         # Raw and processed data
│   ├── raw/                     # Raw dataset
│   └── processed/               # Processed dataset
├── notebooks/                    # Exploratory notebooks
│   └── 1.0-eda.ipynb            # EDA notebook
├── src/                         # Production code
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering
│   ├── train.py                # Model training
│   ├── predict.py              # Inference
│   └── api/
│       ├── main.py             # FastAPI application
│       └── pydantic_models.py  # API data validation
├── tests/                       # Unit tests
│   └── test_data_processing.py  # Data processing tests
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

**Setup**

Clone the repository:
	```bash
	git clone https://github.com/smucav/credit-risk-mlops.git
	cd credit-risk-mlops
	```


Create a virtual environment and install dependencies:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	pip install -r requirements.txt
	```

##  Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The **Basel II Capital Accord** mandates robust risk measurement, transparency, and regulatory compliance for financial institutions issuing loans. It requires banks to **quantify credit risk accurately** and **justify their models to regulators**.

An **interpretable model**, such as **Logistic Regression with Weight of Evidence (WoE) transformations**, is critical because it allows stakeholders to understand how features (e.g., transaction frequency) contribute to risk predictions. Clear documentation of the model’s **assumptions**, **feature engineering**, and **performance metrics** ensures compliance with Basel II’s transparency requirements, facilitating **audits** and **regulatory approval**.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

The dataset lacks a direct "**default**" label, necessitating a **proxy variable** to enable supervised learning. We create a binary `is_high_risk` variable by clustering customers based on **RFM (Recency, Frequency, Monetary)** metrics, labeling less engaged customers (e.g., low frequency, low monetary value) as high-risk. This proxy assumes that disengaged customers are **more likely to default**.

However, potential business risks include:

- **❌ Misclassification**: If RFM patterns do not accurately reflect default behavior, the model may incorrectly approve or deny loans, leading to **financial losses** or **missed opportunities**.
- **⚠️ Bias**: The proxy may overgeneralize, **unfairly penalizing** certain customer segments.
- **🧐 Regulatory Scrutiny**: Regulators may question the proxy’s validity, requiring **robust documentation** to justify its use.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

#### ✅ Simple Models (e.g., Logistic Regression with WoE)

**Advantages:**
- Highly **interpretable**, as coefficients directly show feature impacts.
- Aligns with Basel II’s **transparency requirements**, making it easier to explain to regulators and stakeholders.
- **Computationally efficient** and suitable for smaller datasets.

**Disadvantages:**
- May **underperform** on complex, non-linear patterns in alternative data.
- Lower predictive accuracy in some cases.

#### ⚡ Complex Models (e.g., Gradient Boosting)

**Advantages:**
- Capture **intricate patterns** in data, potentially improving accuracy.
- Handle **non-linear relationships** in RFM features effectively.

**Disadvantages:**
- Less interpretable — feature importance is harder to explain.
- Raises **regulatory concerns** under Basel II, requiring additional explanation tools (e.g., SHAP values).
- **Higher computational cost** and risk of **overfitting**.

In a **regulated financial context**, **interpretability often outweighs marginal performance gains**, favoring simpler models unless complex models are **justified with robust explanations**.

---
