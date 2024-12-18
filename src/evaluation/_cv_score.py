import numpy as np
from sklearn.metrics import mean_absolute_error
def mae_inverse(y_true, y_pred):
    return mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))