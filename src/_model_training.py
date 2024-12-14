# %%
import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_gamma_deviance
from src.data import create_sample_split
from src.feature_engineering import LogTransformer

# %%
input_filepath = r"D:\24fall\DS_Project\src\data\cleaned_dataset.parquet"

df = pd.read_parquet(input_filepath)

y = df["Rented Bike Count"]

create_sample_split(df, "Date", training_frac=0.8)

train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()


categoricals = [
    'Seasons', 
    'Holiday', 
    'Week Status'
    ]
numericals= [
    "Hour",
    "Temperature",
    "Humidity",
    "Wind speed",
    "Visibility",
    "Dew point temperature",
    "Solar Radiation",
    "RainOrSnow",
    ]

predictors = categoricals + numericals

X_train=df_train[predictors]
X_test=df_test[predictors]
y_train= df_train.iloc[:,-2]
y_test= df_test.iloc[:,-2]

# %%
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("spline", SplineTransformer(include_bias=False, knots="quantile")),
                ]
            ),
            numericals,
        ),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)

TweedieDist = TweedieDistribution(2)

model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist, l1_ratio=0.01, alpha=0.01,fit_intercept=True
            ),
        ),
    ]
)


# Train GLM model with splines
model_pipeline.fit(X_train, y_train_log)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_))
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_train["pp_t_glm2"] = np.expm1(model_pipeline.predict(X_train))
df_test["pp_t_glm2"] = np.expm1(model_pipeline.predict(X_test))

y_pred_train = model_pipeline.predict(X_train)
y_pred_test = model_pipeline.predict(X_test)


print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_log, y_pred_train)
    )
)
print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_log, y_pred_test)
    )
)


print(f"Train R²: {r2_score(y_train_log, y_pred_train)}")
print(f"Test R²: {r2_score(y_test_log, y_pred_test)}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train_log, y_pred_train))}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test_log, y_pred_test))}")
print(f"Train MAE: {mean_absolute_error(y_train_log, y_pred_train)}")
print(f"Test MAE: {np.sqrt(mean_squared_error(y_test_log, y_pred_test))}")


# %%
from lightgbm import LGBMRegressor

# Let's use a GBM instead as an estimator

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.fit_transform(X_test)

model_pipeline = Pipeline([("estimate", LGBMRegressor(objective="tweedie"))])

# model_pipeline.fit(X_train_t, y_train)

# df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
# df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
# print(
#     "training loss t_lgbm:  {}".format(
#         TweedieDist.deviance(y_train, df_train["pp_t_lgbm"])
#     )
# )

# print(
#     "testing loss t_lgbm:  {}".format(
#         TweedieDist.deviance(y_test, df_test["pp_t_lgbm"])
#     )
# )


model_pipeline.fit(X_train_t, y_train_log)

df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_log, df_train["pp_t_lgbm"])
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_log, df_test["pp_t_lgbm"])
    )
)

# %%
from sklearn.model_selection import GridSearchCV
# Let's tune the pipeline to reduce overfitting

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned
cv = GridSearchCV(
    model_pipeline,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        "estimate__n_estimators": [50, 100, 150, 200],
    },
    verbose=2,
)
cv.fit(X_train_t, y_train_log)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_log, df_train["pp_t_lgbm"])
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_log, df_test["pp_t_lgbm"])
    )
)


# %%

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 预测结果（假设你已经有了预测值）
y_pred_train = cv.best_estimator_.predict(X_train_t)  # 训练集预测结果
y_pred_test = cv.best_estimator_.predict(X_test_t)    # 测试集预测结果

# 计算训练集和测试集的评估指标
print(f"Train R²: {r2_score(y_train_log, y_pred_train)}")  # 训练集 R²
print(f"Test R²: {r2_score(y_test_log, y_pred_test)}")    # 测试集 R²

# 计算 RMSE（均方根误差）
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train_log, y_pred_train))}")  # 训练集 RMSE
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test_log, y_pred_test))}")    # 测试集 RMSE

# 计算 MAE（平均绝对误差）
print(f"Train MAE: {mean_absolute_error(y_train_log, y_pred_train)}")  # 训练集 MAE
print(f"Test MAE: {mean_absolute_error(y_test_log, y_pred_test)}")    # 测试集 MAE
