import numpy as np
import pandas as pd

from retail_sales import data, model


def _synthetic(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "Age": rng.integers(18, 70, n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Product Category": rng.choice(["Beauty", "Clothing", "Electronics"], n),
            "Total Amount": rng.integers(25, 2000, n),
        }
    )


def test_add_date_features():
    df = data.add_date_features(_synthetic(5))
    assert {"Year", "Month", "Day", "Weekday"} <= set(df.columns)


def test_train_and_predict(tmp_path):
    df = data.add_date_features(_synthetic())
    X_train, X_test, y_train, y_test = model.split(df)
    pipeline = model.build_pipeline().fit(X_train, y_train)

    assert set(model.evaluate(pipeline, X_test, y_test)) == {"mae", "mse", "rmse", "r2"}

    path = tmp_path / "m.joblib"
    model.save(pipeline, path)
    loaded = model.load(path)
    assert isinstance(model.predict_one(loaded, 30, "Female", "Beauty", 6, 15), float)
