import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

class Discretizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to discretize continuous features into categorical bins.
    """
    def __init__(self, n_bins=3, encode='ordinal', strategy='quantile'):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)

    def fit(self, X, y=None):
        """Fit the KBinsDiscretizer on the data."""
        self.discretizer.fit(X)
        return self

    def transform(self, X):
        """Transform the input data using the fitted KBinsDiscretizer."""
        return self.discretizer.transform(X)

class SparseFeatureCombiner(Discretizer):
    """
    Custom transformer to combine sparse features into a single binary feature.
    This transformer extends the Discretizer to handle the binary encoding of multiple sparse features.
    If any of the specified features is non-zero, the combined feature will be 1, otherwise 0.
    
    Parameters:
    -----------
    n_bins : int, default=2
        Number of bins to discretize each feature (default binarization).
    
    encode : str, default='ordinal'
        Encoding method for the Discretizer.
    
    strategy : str, default='uniform'
        Strategy to use for discretizing features.
    
    Methods:
    --------
    transform(X):
        Combine specified features into a single binary feature.
    
    Example:
    --------
    combiner = SparseFeatureCombiner()
    combined_feature = combiner.transform(X)  # X contains multiple sparse columns
    """
    def __init__(self):
        super().__init__(n_bins=2, encode='ordinal', strategy='uniform')  # Binarize the feature

    def transform(self, X):
        """
        Combine sparse features into a binary feature.
        If any feature > 0, output 1, otherwise 0.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input array with multiple columns representing sparse features.
        
        Returns:
        --------
        combined_feature : array, shape (n_samples, 1)
            Binary feature indicating if any of the sparse features are non-zero.
        """
        if isinstance(X, pd.DataFrame):  # If input is a DataFrame, convert to NumPy array
            X = X.values
            
        # Apply the original Discretizer transformation (binarization of each feature)
        X_bin = super().transform(X)
        
        # Combine all sparse features: 1 if any column is non-zero
        combined_feature = (X_bin > 0).any(axis=1).astype(int)
        
        return combined_feature.reshape(-1, 1)