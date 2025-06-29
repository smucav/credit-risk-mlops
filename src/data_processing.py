import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from datetime import datetime
import os

class TimeFeatureExtractor:
    """Class to extract time-based features from TransactionStartTime."""

    def fit(self, X, y=None):
        """Fit method (no fitting needed for transformation)."""
        return self

    def transform(self, X):
        """Transform X by extracting time features."""
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

    def get_feature_names_out(self, input_features=None):
        """Return the names of the transformed features."""
        return ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']

class AggregateFeatureExtractor:
    """Class to extract aggregate features per CustomerId."""

    def __init__(self):
        """Initialize with empty groups."""
        self.groups = None

    def fit(self, X, y=None):
        """Fit by grouping data by CustomerId."""
        self.groups = X.groupby('CustomerId')
        return self

    def transform(self, X):
        """Transform X by adding aggregate features."""
        X = X.copy()
        agg_features = self.groups['Amount'].agg(
            ['sum', 'mean', 'count', 'std']
        ).rename(columns={
            'sum': 'TotalTransactionAmount',
            'mean': 'AverageTransactionAmount',
            'count': 'TransactionCount',
            'std': 'StdTransactionAmount'
        })
        return X.merge(agg_features, on='CustomerId', how='left')

    def get_feature_names_out(self, input_features=None):
        """Return the names of the transformed features."""
        return ['TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount']

class DataProcessor:
    """Class to manage data processing pipeline and execution."""

    def __init__(self, raw_path="data/raw/data.csv", processed_path="../data/processed/processed_data.csv"):
        """Initialize with file paths."""
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.pipeline = self._create_pipeline()

    def _load_data(self):
        """Load raw dataset."""
        return pd.read_csv(self.raw_path)

    def _create_pipeline(self):
        """Create and return the preprocessing pipeline."""
        # Identify column types
        categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'ProductId', 'CurrencyCode']
        numerical_cols = ['Amount', 'Value', 'CountryCode', 'PricingStrategy', 'FraudResult']
        time_col = ['TransactionStartTime']

        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        # Preprocessing for time data
        time_transformer = Pipeline(steps=[
            ('time_extractor', TimeFeatureExtractor())
        ])

        # Aggregate features transformer
        agg_transformer = Pipeline(steps=[
            ('agg_extractor', AggregateFeatureExtractor())
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols),
                ('time', time_transformer, time_col),
                ('agg', agg_transformer, ['CustomerId', 'Amount'])
            ],
            remainder='passthrough'
        )

        # Full pipeline
        return Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])

    def process(self):
        """Process the data and save the result."""
        # Load data
        df = self._load_data()

        # Fit and transform
        processed_data = self.pipeline.fit_transform(df)

        # Get the number of features from the transformed data
        n_features = processed_data.shape[1]
        # Use default integer indices as column names if get_feature_names_out fails
        column_names = [f"feature_{i}" for i in range(n_features)] if n_features != len(self.pipeline.get_feature_names_out()) else self.pipeline.get_feature_names_out()

        # Convert to DataFrame with computed or pipeline-derived column names
        processed_df = pd.DataFrame(processed_data, columns=column_names)
        processed_df.index = df.index

        # Ensure all original columns are included
        for col in df.columns:
            if col not in processed_df.columns:
                processed_df[col] = df[col]

        # Save processed data
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        processed_df.to_csv(self.processed_path, index=False)
        print(f"Processed data saved to {self.processed_path}")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process()
