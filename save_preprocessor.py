#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Load data
df_final = pd.read_csv("data/processed/xente_final_with_target.csv")

# Feats (match your Task 5)
id_cols = [
    "CustomerId",
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "TransactionStartTime",
    "is_high_risk",
]
categorical_cols = [
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
]
feat_cols = [col for col in df_final.columns if col not in id_cols]
num_cols = [col for col in feat_cols if col not in categorical_cols]

print("Num cols (should be 8):", num_cols)
print("Cat cols:", categorical_cols)
print("Total raw feats:", len(num_cols + categorical_cols))

X = df_final[num_cols + categorical_cols]

# Preprocessor (exact as Task 5)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
)

# Fit on full X (matches training data variation)
preprocessor.fit(X)

# Print output feats (should match model's 15 or whatever it is)
feat_names = preprocessor.get_feature_names_out()
print("Output feats count:", len(feat_names))
print("Output feats sample:", feat_names[:5].tolist())  # First 5 for check

# Save
joblib.dump(preprocessor, "data/processed/preprocessor.pkl")
print("Saved preprocessor.pkl—now load in API!")
