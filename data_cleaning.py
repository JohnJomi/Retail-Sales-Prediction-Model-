import glob
import os

import kagglehub
import pandas as pd

# load data
path = kagglehub.dataset_download("mohammadtalib786/retail-sales-dataset")
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
df = pd.read_csv(csv_files[0])

print(df.shape)
print(df.dtypes)
print(df.head())

# clean data
print(df.isnull().sum())
print(df.duplicated().sum())

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()

print(df.describe())
print(df.head())
