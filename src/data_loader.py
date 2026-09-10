
import pandas as pd


def load_data(path: str = "data/car_data.csv") -> pd.DataFrame:
 
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
    df = load_data()
    summarize(df)
