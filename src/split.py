"""
Stage 3: Train/Test Split
---------------------------
Splits the processed DataFrame into train and test sets for the
target column "Selling_Price".
"""

from sklearn.model_selection import train_test_split


def split_data(
    df,
    target: str = "Selling_Price",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split a processed DataFrame into X/Y train and test sets.

    Args:
        df: Fully numeric DataFrame (output of feature_engineering.engineer_features).
        target: Name of the target column to predict.
        test_size: Fraction of data to hold out for testing.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, Y_train, Y_test
    """
    X = df.drop(columns=[target])
    Y = df[target]

    return train_test_split(X, Y, test_size=test_size, random_state=random_state)
