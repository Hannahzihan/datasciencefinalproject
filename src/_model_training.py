#%%
import numpy as np
import pandas as pd
import dalex as dx
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from dask_ml.preprocessing import Categorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer

from src.plotting import plot_predicted_vs_actual, plot_multi_model_lorenz_curve,coefficient_plotting
from src.evaluation import mae_inverse,evaluate_model

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#%%
# Load dataset
input_filepath = r"D:\24fall\DS_Project\src\data\cleaned_dataset.parquet"
df = pd.read_parquet(input_filepath)


# Additional feature engineering

# Create binary feature 'rain_snow' indicating presence of rain or snow
df['rain_snow'] = ((df['snow'] > 0) | (df['rain'] > 0)).astype(int)

# Categorize Solar Radiation and Visibility into three levels
solar_thresholds = [df['sol_rad'].min()-0.01, 0.1, 1.5, df['sol_rad'].max() + 0.001]
visibility_thresholds = [df['visib'].min()-0.01, 500, 1900, df['visib'].max() + 0.001]
df['sol_rad_level'] = pd.cut(df['sol_rad'], bins=solar_thresholds, labels=['Low', 'Medium', 'High'])
df['visib_level'] = pd.cut(df['visib'], bins=visibility_thresholds, labels=['Low', 'Medium', 'High'])

# Split data into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

# Define target variable
target = "bike_cnt"

# Specify categorical and numerical features for the model
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
#drop transformed and less relevant data

# Combine all predictors
predictors = categoricals + numericals

# Prepare training and testing data
X_train = df_train[predictors]
X_test = df_test[predictors]
y_train = df_train[target]
y_test = df_test[target]

# Log-transform the target to reduce skewness
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

#%%
# Set up preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline([
            ('spline', SplineTransformer(n_knots=4, degree=3)),
            ('scaler', StandardScaler())
        ]), numericals),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals)
    ]
)

# Build pipeline with preprocessing and a generalized linear model (GLM)
glm_model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("estimate", GeneralizedLinearRegressor(family=TweedieDistribution(power=1.5), fit_intercept=True))
])

# Define grid of hyperparameters for tuning
glm_param_grid = {
    "estimate__alpha": [0.001, 0.01, 0.1, 1, 10],  # Regularization strengths
    "estimate__l1_ratio": [0, 0.25, 0.5, 0.75, 1]  # ElasticNet mixing ratios
}

# Setup K-fold cross-validation and mean absolute error (MAE) scorer
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scorer = make_scorer(mae_inverse, greater_is_better=False)

# Configure and fit GridSearchCV
glm_grid_search = GridSearchCV(
    estimator=glm_model_pipeline,
    param_grid=glm_param_grid,
    scoring=mae_scorer,
    cv=kf,
    verbose=2,
    n_jobs=-1
)

# Fit the GridSearchCV on training data
glm_grid_search.fit(X_train, y_train_log)

# Print best hyperparameters and corresponding score
print("Best Parameters for GLMs:", glm_grid_search.best_params_)
print("Best MAE Score for GLMs:", -glm_grid_search.best_score_)

# Evaluate model on training and testing sets
glm_y_pred_train, glm_metrics_train = evaluate_model(glm_grid_search, X_train, y_train)
glm_y_pred_test, glm_metrics_test = evaluate_model(glm_grid_search, X_test, y_test)
print("GLM Metrics on Train:", glm_metrics_train)
print("GLM Metrics on Test:", glm_metrics_test)

# Plot predicted vs. actual values
plot_predicted_vs_actual(y_test, glm_y_pred_test, 'GLM Predicted vs Actual')

# Explain model using Dalex and plot the contribution of the top variables
glm_explainer = dx.Explainer(glm_grid_search.best_estimator_, X_train, y_train_log, label="GLM")
glm_explainer.model_parts().plot()

# Plot the coefficient of features
coefficient_plotting(glm_grid_search.best_estimator_, preprocessor, numericals, categoricals)
# %%
# Preprocess categorical features for LightGBM
lgbm_categorizer = Categorizer(columns=categoricals)
X_train_t = lgbm_categorizer.fit_transform(X_train)
X_test_t = lgbm_categorizer.transform(X_test)

# Setup LightGBM pipeline with tweedie objective
lgbm_model_pipeline = Pipeline([("estimate", LGBMRegressor(objective="tweedie"))])

# Define the hyperparameter search grid
lgbm_param_grid = {
    "estimate__learning_rate": [0.01, 0.05, 0.1],
    "estimate__n_estimators": [500, 1000, 1500],
    "estimate__num_leaves": [30,40,50],
    "estimate__min_child_weight": [0.001,0.002, 0.004],
}

# Configure 5-fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Configure RandomizedSearchCV for more efficient search
lgbm_grid_search = RandomizedSearchCV(
    estimator=lgbm_model_pipeline,
    param_distributions=lgbm_param_grid,
    n_iter=30,  # Number of random combinations
    scoring= mae_scorer,
    cv=kf,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit the RandomizedSearchCV on training data
lgbm_grid_search.fit(X_train_t, y_train_log)

# Print the best parameters from hyperparameter tuning
print("Best Parameters for LGBMs:", lgbm_grid_search.best_params_)
print("Best MAE Score for LGBMs:", -lgbm_grid_search.best_score_)

# Evaluate model performance on train and test datasets
lgbm_y_pred_train, lgbm_metrics_train = evaluate_model(lgbm_grid_search, X_train_t, y_train)
lgbm_y_pred_test, lgbm_metrics_test = evaluate_model(lgbm_grid_search, X_test_t, y_test)
print("LGBM Metrics on Train:", lgbm_metrics_train)
print("LGBM Metrics on Test:", lgbm_metrics_test)

# Plot predicted vs. actual results for test dataset
plot_predicted_vs_actual(y_test, lgbm_y_pred_test, 'LGBM Predicted vs Actual')

# Explain model using Dalex and plot the contribution of the top variables
lgbm_explainer = dx.Explainer(lgbm_grid_search.best_estimator_, X_train_t, y_train_log, label="LGBM")
lgbm_explainer.model_parts().plot()

# Calculate SHAP values for LGBM
lgbm_model = lgbm_grid_search.best_estimator_.named_steps['estimate']
lgbm_shap_explainer = shap.TreeExplainer(lgbm_model)
lgbm_shap_values = lgbm_shap_explainer.shap_values(X_test_t)
shap.summary_plot(lgbm_shap_values, X_test_t)

# Rank the split time of each feature
lgbm_model = lgbm_grid_search.best_estimator_.named_steps['estimate']
lgb.plot_importance(lgbm_model, max_num_features=20)  
plt.show()

#%%
# Plot the Lorenz curve 
y_test_array = np.array(y_test)  # Convert y_test to numpy array if not already
glm_predictions = np.array(glm_y_pred_test)  # Same for predictions
lgbm_predictions = np.array(lgbm_y_pred_test)

model_predictions = [glm_predictions, lgbm_predictions]
model_labels = ['GLM Model', 'LGBM Model']

plot_multi_model_lorenz_curve(y_test_array, model_predictions, model_labels)

# %%
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
# Using PartialDependenceDisplay for generating PDPs

numeric_features = ['temp', 'hum','rain_snow']  # Ensure these are numeric in X_train_t_preprocessed
PartialDependenceDisplay.from_estimator(
    lgbm_grid_search.best_estimator_, 
    X_train_t, 
    features=numeric_features,  # Make sure to only include numeric features here
    n_jobs=-1, 
    grid_resolution=50
)
plt.suptitle("Partial Dependence Plots")
plt.subplots_adjust(top=0.9)  # Adjust plot to make room for title
plt.show()
