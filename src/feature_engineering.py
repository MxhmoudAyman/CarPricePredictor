

import pandas as pd

CURRENT_YEAR = 2026
CATEGORICAL_COLUMNS = ["Fuel_Type", "Selling_type", "Transmission"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  
    df = df.copy()

    # Turn absolute Year into a relative Car_Age feature.
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])

    # One-hot encode categorical text columns.
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    return df


if __name__ == "__main__":
    from data_loader import load_data

    raw_df = load_data("../data/car_data.csv")
    processed_df = engineer_features(raw_df)
    print(processed_df.head())
    print(processed_df.dtypes)
