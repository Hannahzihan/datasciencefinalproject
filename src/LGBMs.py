import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# =======================
# 1️⃣ Load and Preprocess Data
# =======================

# Load dataset
input_filepath = r"D:\24fall\DS_Project\src\data\cleaned_dataset.parquet"
df = pd.read_parquet(input_filepath)

# Create a binary feature 'RainOrSnow' which is 1 if either Rainfall or Snowfall is non-zero
df['RainOrSnow'] = ((df['Snowfall'] > 0) | (df['Rainfall'] > 0)).astype(int)

# Additional feature engineering
# Divide Solar Radiation and Visibility into three categories: low, medium, high
# Define manual thresholds for Solar Radiation and Visibility
solar_thresholds = [0, 0.1, 1.5, df['Solar Radiation'].max()]  # Customize these thresholds
visibility_thresholds = [0, 500, 1900, df['Visibility'].max()]  # Customize these thresholds

# Use pd.cut() to categorize Solar Radiation and Visibility
df['SolarRadiation_Level'] = pd.cut(df['Solar Radiation'], bins=solar_thresholds, labels=['Low', 'Medium', 'High'])
df['Visibility_Level'] = pd.cut(df['Visibility'], bins=visibility_thresholds, labels=['Low', 'Medium', 'High'])

df['Log_Windspeed'] = np.log1p(df['Wind speed'])  # Log transform if skewed

df['Hour'] = df['Hour'].astype(str)
# Split the dataset into training and testing sets (75% train, 25% test)
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

# =======================
# 2️⃣ Define Features and Target Variables
# =======================

target = "Rented Bike Count"  # Target variable

# Categorical and numerical features used for model training
categoricals = [
    'Hour',
    'Seasons', 
    'SolarRadiation_Level',
    'Visibility_Level',
    'Month', 
    "Day of Week",
    'Holiday', 
    'Week Status'
    ]

numericals = [
    # 'Hour',
    "Temperature",
    "Humidity",
    # "Log_Visibility",
    # "Log_SolarRadiation",
    "RainOrSnow",
    #"Dew Point Temperature"
    #"Log_Windspeed",
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

# Create LGBM pipeline
model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("estimate", LGBMRegressor(objective="regression", random_state=42))
])

# =======================
# 4️⃣ Hyperparameter Tuning Setup
# =======================

# Define the hyperparameter search grid
param_grid = {
    "estimate__learning_rate": [0.01, 0.05, 0.1],
    "estimate__n_estimators": [500, 1000, 1500],
    "estimate__num_leaves": [30,40,50],
    "estimate__min_child_weight": [0.001,0.002, 0.004],
}

# Setup cross-validation and early stopping
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Configure RandomizedSearchCV for more efficient search
grid_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_grid,
    n_iter=30,  # Number of random combinations
    scoring='neg_mean_absolute_error',  # Using MAE
    cv=kf,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# =======================
# 5️⃣ Model Training and Hyperparameter Tuning
# =======================

# Fit the RandomizedSearchCV on training data
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