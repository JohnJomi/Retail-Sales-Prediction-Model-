from . import data, model


def main() -> None:
    df = data.add_date_features(data.load_raw())
    X_train, X_test, y_train, y_test = model.split(df)

    pipeline = model.build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = model.evaluate(pipeline, X_test, y_test)
    print("=== Model Metrics ===")
    print(f"MAE:  ${metrics['mae']:.2f}")
    print(f"MSE:  {metrics['mse']:.2f}")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"R2:   {metrics['r2']:.4f}")

    model.save(pipeline)


if __name__ == "__main__":
    main()
