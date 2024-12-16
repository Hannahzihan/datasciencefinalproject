import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply log transformation to the selected features.
    Uses log1p(x) = log(x + 1) to handle zero values.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """No fitting is required for this transformer."""
        return self

    def transform(self, X):
        """Apply log1p transformation to the input features."""
        if isinstance(X, pd.DataFrame):  
            X = X.values
        if X.size == 0:  
            raise ValueError("Input DataFrame must have at least one sample.")
        if np.any(X < 0):  
            raise ValueError("Input contains negative values, which cannot be transformed using log1p.")
        return np.log1p(X)
