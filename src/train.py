"""
Stage 4: Model Training
--------------------------
Trains the candidate models used to predict Selling_Price.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_models(X_train, Y_train, random_state: int = 42) -> dict:
    """Train a Linear Regression and a Random Forest model.

    Args:
        X_train: Training features.
        Y_train: Training targets.
        random_state: Seed for the Random Forest for reproducibility.

    Returns:
        Dict mapping model name -> fitted model instance.
    """
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, Y_train)

    return {"Linear Regression": lr, "Random Forest": rf}
