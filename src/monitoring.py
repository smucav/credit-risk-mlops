#!/usr/bin/env python3

import pandas as pd
import logging
import mlflow
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load training data (assumed to be the same as used in train.py)
train_data_path = "data/processed/processed_data_with_target.csv"
try:
    train_df = pd.read_csv(train_data_path)
    logger.info(f"Loaded training data from {train_data_path}")
except FileNotFoundError:
    logger.error(f"Training data file not found at {train_data_path}")
    raise

# Simulate inference data (e.g., a subset or new sample data)
# For demonstration, use a random subset of training data with slight modifications
inference_df = train_df.sample(frac=0.1, random_state=42).copy()
inference_df["num__Amount"] = inference_df["num__Amount"] * 1.1  # Simulate drift

# Prepare data for drift analysis (drop target and non-numeric columns)
reference_data = train_df.drop(columns=["is_high_risk", "remainder__TransactionId", "remainder__BatchId", "remainder__AccountId", "remainder__SubscriptionId"])
current_data = inference_df.drop(columns=["is_high_risk", "remainder__TransactionId", "remainder__BatchId", "remainder__AccountId", "remainder__SubscriptionId"])

# Initialize Evidently AI report
drift_report = Report(metrics=[
    DataDriftTable(),
    DatasetDriftMetric()
])

# Calculate drift
drift_report.run(reference_data=reference_data, current_data=current_data)
drift_result = drift_report.as_dict()

# Log drift results
logger.info("Drift Report: %s", drift_result)
with mlflow.start_run(run_name="Drift_Monitoring"):
    mlflow.log_text(str(drift_result), "drift_report.json")
    drift_detected = drift_result["metrics"][1]["result"]["dataset_drift"]
    logger.info(f"Dataset drift detected: {drift_detected}")
    if drift_detected:
        logger.warning("Data drift detected! Consider retraining the model.")

# Optional: Save report to file for review
drift_report.save_html("drift_report.html")
logger.info("Drift report saved as drift_report.html")
