import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import make_scorer

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