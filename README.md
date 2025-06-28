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
