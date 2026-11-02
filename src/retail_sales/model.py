import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), config.CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
                ),
            ),
        ]
    )


def split(df: pd.DataFrame):
    return train_test_split(
        df[config.FEATURES],
        df[config.TARGET],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_test, y_pred),
    }


def save(pipeline: Pipeline, path=config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load(path=config.MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_one(pipeline: Pipeline, age, gender, category, month, day) -> float:
    row = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Product Category": [category],
            "Month": [int(month)],
            "Day": [int(day)],
        }
    )
    return float(pipeline.predict(row)[0])
