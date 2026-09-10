from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(models: dict, X_test, Y_test) -> dict:
  
    results = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        results[name] = {
            "r2": r2_score(Y_test, pred),
            "mae": mean_absolute_error(Y_test, pred),
            "mse": mean_squared_error(Y_test, pred),
        }

    return results


def print_results(results: dict) -> None:
   
    for name, metrics in results.items():
        print(f"--- {name} ---")
        print(f"R2 Score: {metrics['r2']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}\n")
