from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Log Transformer for handling skewed data, transforms each feature using log(1 + X).
    The log(1 + X) transformation is used to avoid log(0) when data contains zero values.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting necessary, just returns self.
        return self

    def transform(self, X):
        # Check for non-positive values, which can't be transformed with log.
        if np.any(X < 0):
           raise ValueError("LogTransformer only accepts positive values. Please check your data.")
        
        # Apply log(1 + X) transformation to avoid log(0)
        X_transformed = np.log1p(X)
        
        # Ensure that the result is returned as a pandas DataFrame
        return pd.DataFrame(X_transformed, columns=X.columns)  # Assuming X is a DataFrame
