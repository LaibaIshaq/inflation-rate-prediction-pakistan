"""Train a Lasso Regression model to predict Pakistan inflation.

The script expects the dataset included in ``data/inflation_data.csv``.
It trains a LassoCV model, prints evaluation metrics, and can either prompt for
new economic indicator values or run a sample prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]"
FEATURE_COLUMNS = [
    "Manufacturing, value added (% of GDP) [NV.IND.MANF.ZS]",
    "Foreign direct investment, net inflows (% of GDP) [BX.KLT.DINV.WD.GD.ZS]",
    "Food imports (% of merchandise imports) [TM.VAL.FOOD.ZS.UN]",
    "Official exchange rate (LCU per US$, period average) [PA.NUS.FCRF]",
    "Military expenditure (% of GDP) [MS.MIL.XPND.GD.ZS]",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the inflation dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path).dropna()
    else:
        data = pd.read_csv(path).dropna()

    missing_columns = [col for col in [TARGET_COLUMN, *FEATURE_COLUMNS] if col not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data


def train_model(data: pd.DataFrame):
    """Scale features, split the data, and train the LassoCV model."""
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    x_train, x_test, y_train, y_test = train_test_split(
        scaled_features,
        target.values,
        test_size=0.2,
        random_state=123,
    )

    model = LassoCV(alphas=np.logspace(-6, 2, 100), cv=5)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    return model, scaler, features, metrics


def prompt_for_values(features: pd.DataFrame) -> pd.DataFrame:
    """Collect user inputs within the observed dataset ranges."""
    print("\nEnter values for prediction:\n")
    user_inputs = []

    for column in features.columns:
        min_value = features[column].min()
        max_value = features[column].max()

        while True:
            try:
                value = float(input(f"{column} ({min_value:.2f} to {max_value:.2f}): "))
            except ValueError:
                print("Invalid input. Enter a number.")
                continue

            if min_value <= value <= max_value:
                user_inputs.append(value)
                break

            print("Value out of range. Try again.")

    return pd.DataFrame([user_inputs], columns=features.columns)


def sample_values(features: pd.DataFrame) -> pd.DataFrame:
    """Use the newest row in the dataset as a reproducible sample input."""
    return pd.DataFrame([features.iloc[0].values], columns=features.columns)


def predict_inflation(model: LassoCV, scaler: StandardScaler, values: pd.DataFrame) -> float:
    """Predict inflation for one row of feature values."""
    scaled_values = scaler.transform(values)
    return float(model.predict(scaled_values)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pakistan inflation prediction using Lasso Regression.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/inflation_data.csv"),
        help="Path to the CSV or Excel dataset.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run a sample prediction instead of asking for console input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    model, scaler, features, metrics = train_model(data)

    print("\nModel Trained Successfully (Lasso Regression)")
    print("Best Alpha:", model.alpha_)
    print("\nModel Performance:")
    print("MSE:", metrics["mse"])
    print("R2 Score:", metrics["r2"])

    user_values = sample_values(features) if args.sample else prompt_for_values(features)
    prediction = predict_inflation(model, scaler, user_values)

    print("\n---------------------------------")
    print(f"Predicted Inflation: {prediction:.4f}")
    print("---------------------------------")


if __name__ == "__main__":
    main()
