import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer
# =======================
# 1️⃣ Load and Preprocess Data
# =======================

# Load dataset
input_filepath = r"D:\24fall\DS_Project\src\data\cleaned_dataset.parquet"
df = pd.read_parquet(input_filepath)

# Create a binary feature 'RainOrSnow' which is 1 if either Rainfall or Snowfall is non-zero
df['rain_snow'] = ((df['snow'] > 0) | (df['rain'] > 0)).astype(int)

# Additional feature engineering
# Divide Solar Radiation and Visibility into three categories: low, medium, high
# Define manual thresholds for Solar Radiation and Visibility
solar_thresholds = [df['sol_rad'].min()-0.01, 0.1, 1.5, df['sol_rad'].max() + 0.001]
visibility_thresholds = [df['visib'].min()-0.01, 500, 1900, df['visib'].max() + 0.001]
# Use pd.cut() to categorize Solar Radiation and Visibility
df['sol_rad_level'] = pd.cut(df['sol_rad'], bins=solar_thresholds, labels=['Low', 'Medium', 'High'])
df['visib_level'] = pd.cut(df['visib'], bins=visibility_thresholds, labels=['Low', 'Medium', 'High'])

# Split the dataset into training and testing sets (75% train, 25% test)
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

# =======================
# 2️⃣ Define Features and Target Variables
# =======================

target = "bike_cnt"  # Target variable

# Categorical and numerical features used for model training
categoricals = [
    'hour',
    'season', 
    'sol_rad_level',
    'visib_level',
    'month', 
    "day_of_week",
    'holiday', 
    'week_status'
    ]

numericals = [
    "temp",
    "hum",
    # "visib",
    # "sol_rad",
    # "rain",
    # "snow",
    "rain_snow",
    #"dew_temp"
    #"log_wspd",
]

# Combine all predictors
predictors = categoricals + numericals

# Extract predictors and target for training and testing
X_train = df_train[predictors]
X_test = df_test[predictors]
y_train = df_train[target]
y_test = df_test[target]

# Log-transform the target to reduce skewness
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# =======================
# 3️⃣ Define Preprocessing and Model Pipeline
# =======================

# ColumnTransformer for preprocessing (numerical and categorical features)
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numericals),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals)
    ]
)

# Create GLM pipeline
model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("estimate", GeneralizedLinearRegressor(family=TweedieDistribution(power=1.5), fit_intercept=True))
])

# =======================
# 4️⃣ Hyperparameter Tuning Setup
# =======================

# Custom scorer to evaluate MAE in original space
def mae_inverse(y_true, y_pred):
    return mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))

# Define the hyperparameter search grid
param_grid = {
    "estimate__alpha": [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    "estimate__l1_ratio": [0, 0.25, 0.5, 0.75, 1],  # ElasticNet mixing ratio
}

# Setup cross-validation and scorer
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scorer = make_scorer(mae_inverse, greater_is_better=False)

# Configure GridSearchCV with model pipeline and hyperparameter grid
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring=mae_scorer,
    cv=kf,
    verbose=2,
    n_jobs=-1
)

# =======================
# 5️⃣ Model Training and Hyperparameter Tuning
# =======================

# Fit the GridSearchCV on training data
grid_search.fit(X_train, y_train_log)

# Print the best parameters from hyperparameter tuning
print("Best Parameters:", grid_search.best_params_)
print("Best MAE Score (log-transformed):", -grid_search.best_score_)

# =======================
# 6️⃣ Model Evaluation on Train and Test Sets
# =======================

# Predict on training and testing sets
y_pred_train_log = grid_search.predict(X_train)
y_pred_test_log = grid_search.predict(X_test)

# Convert back to original space
y_pred_train = np.expm1(y_pred_train_log)
y_pred_test = np.expm1(y_pred_test_log)

# Calculate evaluation metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print evaluation metrics for training and testing sets
print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Train R²: {train_r2}")
print(f"Test R²: {test_r2}")