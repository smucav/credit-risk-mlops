# Credit Risk Probability Model for Alternative Data

## Overview
This project develops an end-to-end credit risk scoring model, enabling a buy-now-pay-later service in partnership with an eCommerce platform. Using alternative data (transactional and behavioral data), the model predicts the probability of customer default, assigns credit scores, and suggests optimal loan amounts and durations. The project leverages RFM (Recency, Frequency, Monetary) metrics to create a proxy for credit risk and follows MLOps practices for reproducibility and deployment.

## 📁 Project Structure
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
