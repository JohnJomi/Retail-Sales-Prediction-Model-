# Retail Sales Prediction Model

Predicts the `Total Amount` of a retail transaction from customer and purchase details using a scikit-learn Random Forest pipeline, with a Gradio web UI for interactive predictions.

Dataset: [Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset) (Kaggle), downloaded automatically via `kagglehub`.

## Model

- **Target:** `Total Amount`
- **Features:** `Age`, `Gender`, `Product Category`, `Month`, `Day`
- **Algorithm:** `RandomForestRegressor` (100 trees) behind a `ColumnTransformer` with one-hot encoding
- **Metrics:** MAE, MSE, RMSE, R²

`Quantity` and `Price per Unit` are deliberately excluded. `Total Amount = Quantity x Price per Unit`, so including them would let the model learn arithmetic rather than customer and product patterns.

## Project Structure

```
.
├── data/
│   ├── raw/                 # Downloaded source data (git-ignored)
│   └── processed/           # Cleaned data (git-ignored)
├── models/                  # Trained pipeline artifacts (git-ignored)
├── notebooks/
│   └── retail_sales_prediction.ipynb   # Exploration, EDA, and prototyping
├── reports/figures/         # Generated plots
├── src/retail_sales/
│   ├── config.py            # Paths, feature lists, hyperparameters
│   ├── data.py              # Dataset download, loading, date features
│   ├── model.py             # Pipeline, evaluation, save/load, prediction
│   ├── train.py             # Training entry point
│   └── app.py               # Gradio UI
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Train

```bash
python -m retail_sales.train
```

Prints test-set metrics and saves the pipeline to `models/retail_sales_pipeline.joblib`.

### Run the app

```bash
python -m retail_sales.app
```

Opens a Gradio interface at http://127.0.0.1:7860. Train the model first.

### Test

```bash
pytest
```

## Notebook

`notebooks/retail_sales_prediction.ipynb` contains the original end-to-end workflow: data loading, cleaning, EDA, modeling, evaluation, and deployment. It can also be run in Google Colab.

## Branches

- `main`: stable
- `Dev`: active development
