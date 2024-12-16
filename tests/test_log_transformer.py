import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import LogTransformer

def test_log_transformer_basic():
    """Test LogTransformer with positive numeric values."""
    X = pd.DataFrame({'Feature1': [0, 1, 10, 100]})
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    expected = np.log1p(X)
    np.testing.assert_array_almost_equal(X_transformed, expected.values, decimal=6)  # Allow slight floating-point error

def test_log_transformer_zeros():
    """Test LogTransformer to correctly handle zeros."""
    X = pd.DataFrame({'Feature1': [0, 0, 0, 0]})
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    assert np.all(X_transformed == 0)  # log1p(0) = 0

def test_log_transformer_negative_values():
    """Test LogTransformer to raise an error when negative values are present."""
    X = pd.DataFrame({'Feature1': [-1, -2, -3]})
    transformer = LogTransformer()
    with pytest.raises(ValueError, match="negative values"):
        transformer.fit_transform(X)

def test_log_transformer_non_numeric():
    """Test LogTransformer to raise an error for non-numeric data."""
    X = pd.DataFrame({'Feature1': ['a', 'b', 'c']})
    transformer = LogTransformer()
    with pytest.raises(TypeError):
        transformer.fit_transform(X)

def test_log_transformer_empty_dataframe():
    """Test LogTransformer with an empty DataFrame."""
    X = pd.DataFrame(columns=['Feature1'])
    transformer = LogTransformer()
    with pytest.raises(ValueError, match="at least one sample"):
        transformer.fit_transform(X)

def test_log_transformer_output_shape():
    """Test that LogTransformer output shape matches input shape."""
    X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [10, 20, 30]})
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    assert X_transformed.shape == X.shape

def test_log_transformer_supports_numpy_array():
    """Test that LogTransformer works with NumPy arrays."""
    X = np.array([[0, 1, 2], [3, 4, 5]])
    transformer = LogTransformer()
    X_transformed = transformer.fit_transform(X)
    expected = np.log1p(X)
    np.testing.assert_array_almost_equal(X_transformed, expected, decimal=6)  # Allow slight floating-point error
