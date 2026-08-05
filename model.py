"""Generate a demo LogisticRegression model + test CSV (UCI Adult Income)."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

print("Downloading UCI Adult Income dataset...")
data = pd.read_csv(
    url, header=None, names=columns, na_values=" ?", skipinitialspace=True
)
data.dropna(inplace=True)

X = data.drop("income", axis=1)
y = data["income"].apply(lambda x: 1 if x == ">50K" else 0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns
numerical_features = X.select_dtypes(include=["number"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

print("Training LogisticRegression pipeline...")
model_pipeline.fit(X_train, y_train)

model_path = ROOT / "model.pkl"
csv_path = ROOT / "test_data.csv"
joblib.dump(model_pipeline, model_path)
X_test = X_test.copy()
X_test["income"] = y_test
X_test.to_csv(csv_path, index=False)
print(f"Saved {model_path.name} and {csv_path.name}")
