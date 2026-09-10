"""
Stage 1: Data Loading
----------------------
Loads the raw car dataset from CSV and reports basic sanity info.
"""

import pandas as pd


def load_data(path: str = "data/car_data.csv") -> pd.DataFrame:
    """Load the raw car dataset from a CSV file.

    Args:
        path: Path to the car_data.csv file.

    Returns:
        Raw DataFrame, unmodified.
    """
    df = pd.read_csv(path)
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print a quick sanity check of the loaded data."""
    print(f"Shape: {df.shape}")
    print("\nFirst rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    # Allows running this stage standalone: python src/data_loader.py
    df = load_data()
    summarize(df)
