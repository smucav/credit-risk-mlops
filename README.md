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
|   ├── target_engineering.py  # Proxy target variable engineering
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

## 2: Exploratory Data Analysis (EDA)

#### Objective
Perform an exploratory analysis of the Xente dataset (`data/raw/data.csv`) to understand its structure, distributions, correlations, missing values, and outliers, providing insights for feature engineering.

#### Implementation
- **Notebook**: `notebooks/1.0-eda.ipynb`
  - Loads the dataset with 95,662 rows and 16 columns.
  - Analyzes data types (11 object, 4 int64, 1 float64) and confirms no missing values.
  - Computes summary statistics: `Amount` (mean 6,717.85, range -1M to 9.88M), `Value` (mean 9,900.58, range 2 to 9.88M), `FraudResult` (mean 0.002, ~0.2% fraud).
  - Visualizes distributions (highly skewed `Amount` and `Value`), box plots (significant outliers), and categorical frequencies (`ProductCategory`, `ChannelId`).
  - Calculates correlations: `Amount`-`Value` (0.99), `Amount`-`FraudResult` (0.56).
- **Key Insights**:
    1. **Highly Skewed Numerical Features with Negative Values**: `Amount` and `Value` are right-skewed with negatives (e.g., -1M), suggesting log-transformation after handling debits/credits.
    2. **Dominant Product Categories**: “financial_services” and “airtime” dominate, recommending one-hot encoding with rare category grouping.
    3. **No Missing Data**: All rows are complete, eliminating imputation needs.
    4. **Complementary but Highly Correlated Amount and Value**: Correlation of 0.99, with `Value` as absolute size and `Amount` including direction; derive debit/credit flags.
    5. **Significant Outliers**: Extremes from -1M to 9.88M, suggesting capping at 1.5× IQR.


## 3: Feature Engineering

#### Objective
Build a robust, automated, and reproducible data processing script to transform raw data into a model-ready format using an OOP design.

#### Implementation
- **Script**: `src/data_processing.py`
  - Uses a `DataProcessor` class with `sklearn.pipeline.Pipeline` to chain transformations.
  - **Aggregate Features**:
    - `TotalTransactionAmount`: Sum of `Amount` per `CustomerId`.
    - `AverageTransactionAmount`: Mean `Amount` per `CustomerId`.
    - `TransactionCount`: Number of transactions per `CustomerId`.
    - `StdTransactionAmount`: Standard deviation of `Amount` per `CustomerId`.
  - **Extracted Features**:
    - `TransactionHour`, `TransactionDay`, `TransactionMonth`, `TransactionYear` from `TransactionStartTime`.
  - **Categorical Encoding**:
    - One-hot encoding for `ProductCategory`, `ChannelId`, `ProviderId`, `ProductId`, `CurrencyCode`.
  - **Missing Values**: No imputation needed (confirmed in Task 2), but pipeline includes median imputation for robustness.
  - **Normalization/Standardization**: `StandardScaler` for numerical features (`Amount`, `Value`, etc.).
  - Saves processed data to `data/processed/processed_data.csv`.


## 4 - Proxy Target Variable Engineering

### 📝 Description
Since no pre-existing `"credit risk"` column exists in the dataset, a **proxy target variable** `is_high_risk` was engineered in `src/target_engineering.py` to identify **disengaged customers** as high-risk proxies. The process included:

---

### 📊 RFM Metrics
- **Recency**: Days since the last transaction (calculated using snapshot date **June 30, 2025**).
- **Frequency**: Total number of transactions per `CustomerId`.
- **Monetary**: Total transaction value per `CustomerId`.

These were derived from the raw data using groupby-aggregation.

---

### 📈 Clustering
- Applied **K-Means clustering** on the **scaled RFM features**
- Parameters: `n_clusters=3`, `random_state=42`
- Cluster centers were analyzed to segment customers into behavioral groups.

---

### 🚨 High-Risk Label Assignment
- Calculated an **engagement score** using:
  `engagement = Frequency + |Monetary|`
- The cluster with the **lowest engagement** (i.e., high Recency, low Frequency, low Monetary) was labeled as **high risk**.
- Assigned:
  - `is_high_risk = 1` for customers in the least engaged cluster
  - `is_high_risk = 0` for others

---

### 🔗 Integration
- Merged `is_high_risk` back into the original processed dataset using `CustomerId`
- Saved the updated dataset as:
  `data/processed/processed_data_with_target.csv`

---

### 📁 Output
- Final dataset includes:
  - All **55 features** from Task 3
  - Plus the **new binary target column**: `is_high_risk`
