

from sklearn.model_selection import train_test_split


def split_data(
    df,
    target: str = "Selling_Price",
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=[target])
    Y = df[target]

    return train_test_split(X, Y, test_size=test_size, random_state=random_state)
