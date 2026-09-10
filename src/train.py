
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_models(X_train, Y_train, random_state: int = 42) -> dict:
   
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, Y_train)

    return {"Linear Regression": lr, "Random Forest": rf}
