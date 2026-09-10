from src.data_loader import load_data, summarize
from src.feature_engineering import engineer_features
from src.split import split_data
from src.train import train_models
from src.evaluate import evaluate_models, print_results
from src.visualize import plot_results

DATA_PATH = "data/car_data.csv"
PLOT_PATH = "outputs/result_plots.png"


def main():
    # Stage 1: Load
    print("Stage 1/6: Loading data...")
    df = load_data(DATA_PATH)
    summarize(df)

    # Stage 2: Feature engineering
    print("\nStage 2/6: Engineering features...")
    df = engineer_features(df)

    # Stage 3: Split
    print("\nStage 3/6: Splitting train/test...")
    X_train, X_test, Y_train, Y_test = split_data(df)

    # Stage 4: Train
    print("\nStage 4/6: Training models...")
    models = train_models(X_train, Y_train)

    # Stage 5: Evaluate
    print("\nStage 5/6: Evaluating models...")
    results = evaluate_models(models, X_test, Y_test)
    print_results(results)

    # Stage 6: Visualize (Random Forest, since it has feature importances)
    print("Stage 6/6: Generating plots...")
    plot_results(models["Random Forest"], X_train, X_test, Y_test, PLOT_PATH)

    print("\nDone. See outputs/result_plots.png")


if __name__ == "__main__":
    main()
