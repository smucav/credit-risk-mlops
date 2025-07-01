import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processing import TimeFeatureExtractor


def test_validate_time_range_valid():
    """Test validate_time_range with valid time data."""
    data = pd.DataFrame(
        {"TransactionStartTime": ["2023-01-01 10:00:00", "2023-01-15 15:30:00"]}
    )
    extractor = TimeFeatureExtractor()
    assert extractor.validate_time_range(data) is True


def test_validate_time_range_invalid_hour():
    """Test validate_time_range with invalid hour data."""
    data = pd.DataFrame(
        {"TransactionStartTime": ["2023-01-01 25:00:00"]}  # Invalid hour
    )
    extractor = TimeFeatureExtractor()
    try:
        extractor.validate_time_range(data)
        assert False, "Should raise ValueError for invalid hour"
    except ValueError:
        assert True


if __name__ == "__main__":
    test_validate_time_range_valid()
    test_validate_time_range_invalid_hour()
