"""Training pipeline for developer stress prediction model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data
from src.models.trainer import ModelTrainer


def main() -> None:
    """Run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train stress prediction model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/developer_stress.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/stress_model.joblib",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="models/metrics.json",
        help="Path to save training metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Developer Stress Prediction - Model Training")
    print("=" * 60)

    print(f"\nLoading data from: {args.data_path}")
    df = load_data(args.data_path)
    print(f"Loaded {len(df)} records")

    print("\nInitializing trainer...")
    trainer = ModelTrainer()

    print(f"\nPreparing data (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=args.test_size)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("\nTraining model...")
    trainer.train(X_train, y_train)
    print("Training complete!")

    print("\nEvaluating model...")
    metrics = trainer.evaluate(X_train, X_test, y_train, y_test)

    print("\n" + "-" * 40)
    print("Training Metrics:")
    print("-" * 40)
    print(f"  Train R²:   {metrics['train_r2']:.4f}")
    print(f"  Train MSE:  {metrics['train_mse']:.2f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
    print("-" * 40)
    print("Test Metrics:")
    print("-" * 40)
    print(f"  Test R²:    {metrics['test_r2']:.4f}")
    print(f"  Test MSE:   {metrics['test_mse']:.2f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.2f}")
    print("-" * 40)

    print("\nPerforming cross-validation...")
    cv_metrics = trainer.cross_validate(X_train, y_train)
    print(f"CV R² Mean: {cv_metrics['cv_r2_mean']:.4f} (+/- {cv_metrics['cv_r2_std']:.4f})")

    print("\nFeature Importance:")
    print("-" * 40)
    importance = trainer.get_feature_importance()
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 50)
        print(f"  {feature:20s} {score:.4f} {bar}")

    print(f"\nSaving model to: {args.output_path}")
    trainer.save_model(args.output_path)

    all_metrics = {**metrics, **cv_metrics, "feature_importance": importance}

    print(f"Saving metrics to: {args.metrics_path}")
    with open(args.metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)

    if metrics["test_r2"] < 0.8:
        print("\nWARNING: Model R² is below 0.8. Consider:")
        print("  - Collecting more training data")
        print("  - Feature engineering")
        print("  - Hyperparameter tuning")


if __name__ == "__main__":
    main()
