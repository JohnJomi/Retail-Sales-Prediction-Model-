from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "retail_sales_pipeline.joblib"

KAGGLE_DATASET = "mohammadtalib786/retail-sales-dataset"

TARGET = "Total Amount"
CATEGORICAL_FEATURES = ["Gender", "Product Category"]
NUMERICAL_FEATURES = ["Age", "Month", "Day"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
