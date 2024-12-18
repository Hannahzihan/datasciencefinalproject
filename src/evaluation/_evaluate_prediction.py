import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def evaluate_model(model, X, y_true):
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)  # inverse of log1p
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return y_pred, {'MAE': mae, 'MSE': mse, 'R2': r2}