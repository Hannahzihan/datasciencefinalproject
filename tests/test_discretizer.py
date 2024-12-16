import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import Discretizer, SparseFeatureCombiner

@pytest.mark.parametrize("n_bins, strategy", [(3, 'quantile'), (5, 'uniform')])
def test_discretizer_bins(n_bins, strategy):
    """Test Discretizer to correctly bin continuous values into n_bins categories."""
    X = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    discretizer = Discretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    X_transformed = discretizer.fit_transform(X)
    assert X_transformed.shape == X.shape  # Shape should remain the same
    assert np.all(X_transformed >= 0)  # Bins should be non-negative
    assert np.all(X_transformed < n_bins)  # Bins should be less than n_bins

def test_discretizer_zero_variance():
    """Test Discretizer with zero-variance (constant) column."""
    X = pd.DataFrame({'Feature1': [5, 5, 5, 5, 5]})
    discretizer = Discretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X_transformed = discretizer.fit_transform(X)
    assert X_transformed.shape == X.shape
    assert np.all(X_transformed == 0)  # All rows will belong to the same bin

def test_sparse_feature_combiner():
    """Test SparseFeatureCombiner to correctly combine Rainfall and Snowfall into binary values."""
    X = pd.DataFrame({'Rainfall': [0, 2, 0, 0, 3.5], 'Snowfall': [0, 0, 0.5, 0, 0]})
    combiner = SparseFeatureCombiner()
    combined_feature = combiner.fit_transform(X)
    expected_output = np.array([[0], [1], [1], [0], [1]])
    np.testing.assert_array_equal(combined_feature, expected_output)

def test_discretizer_empty_dataframe():
    """Test Discretizer with an empty DataFrame."""
    X = pd.DataFrame(columns=['Feature1'])
    discretizer = Discretizer(n_bins=3, encode='ordinal', strategy='uniform')
    with pytest.raises(ValueError):
        discretizer.fit_transform(X)

def test_sparse_feature_combiner_empty_dataframe():
    """Test SparseFeatureCombiner with an empty DataFrame."""
    X = pd.DataFrame(columns=['Rainfall', 'Snowfall'])
    combiner = SparseFeatureCombiner()
    with pytest.raises(ValueError):
        combiner.fit_transform(X)
