import pytest
import pandas as pd
from sklearn.metrics import roc_auc_score


def test_model_auc(df_final):
    y = df_final["is_high_risk"]
    # Mock pred_proba
    pred_proba = np.random.rand(len(y))
    auc = roc_auc_score(y, pred_proba)
    assert auc > 0.5, "Mock AUC below baseline"


def test_feature_count(df_final):
    feat_cols = [
        col for col in df_final.columns if col not in ["CustomerId", "is_high_risk"]
    ]
    assert len(feat_cols) >= 10, "Too few features"
