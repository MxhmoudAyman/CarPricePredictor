"""
Stage 6: Visualization
--------------------------
Produces a two-panel figure:
  1. Predicted vs Actual selling price (for a given model).
  2. Feature importance (only meaningful for tree-based models
     like Random Forest).
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(
    model,
    X_train,
    X_test,
    Y_test,
    save_path: str = "outputs/result_plots.png",
) -> None:
    """Plot predicted-vs-actual and feature importance, and save to disk.

    Args:
        model: A fitted model exposing .predict() and, ideally,
            .feature_importances_ (e.g. RandomForestRegressor).
        X_train: Training features (used for the feature-importance labels).
        X_test: Test features.
        Y_test: Test targets.
        save_path: Where to save the resulting PNG.
    """
    pred = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(Y_test, pred, alpha=0.6, color="#2563eb")
    axes[0].plot(
        [Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--"
    )
    axes[0].set_xlabel("Actual Price (lakh)")
    axes[0].set_ylabel("Predicted Price (lakh)")
    axes[0].set_title("Predicted vs Actual")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values()
        axes[1].barh(importances.index, importances.values, color="#16a34a")
        axes[1].set_title("Feature Importance")
    else:
        axes[1].axis("off")
        axes[1].set_title("Feature importance not available for this model")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")
