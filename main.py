"""
Main entry point for training, evaluating, and preparing data
for the churn prediction model.

This script provides CLI commands to:
- prepare_data: load and split dataset
- train: train a new model and log to MLflow
- evaluate: evaluate a saved model and log to MLflow

It integrates MLflow for experiment tracking.
"""

import argparse
import mlflow
import mlflow.sklearn
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# Configure MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ChurnPrediction")


def main(action):
    """
    Execute the pipeline based on the specified action.

    Args:
        action (str): One of 'prepare_data', 'train', or 'evaluate'.
    """
    if action == "prepare_data":
        x_train, x_test, y_train, y_test = prepare_data()
        print("✅ Data prepared successfully")
        print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    elif action == "train":
        x_train, x_test, y_train, y_test = prepare_data()

        with mlflow.start_run(run_name="ChurnModelTraining", tags={"stage": "training"}):
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)

            # Train model
            model = train_model(x_train, y_train)

            # Evaluate model
            accuracy = evaluate_model(model, x_test, y_test)
            print(f"✅ Model trained with accuracy: {accuracy:.4f}")

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save locally
            save_model(model)

    elif action == "evaluate":
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        accuracy = evaluate_model(model, x_test, y_test)
        print(f"📊 Loaded model accuracy: {accuracy:.4f}")

        with mlflow.start_run(run_name="ChurnModelEvaluation", tags={"stage": "evaluation"}):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")

    else:
        print("⚠️ Unknown action. Use 'prepare_data', 'train', or 'evaluate'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Churn prediction pipeline with MLflow tracking."
    )
    parser.add_argument(
        "action",
        choices=["prepare_data", "train", "evaluate"],
        help="Action to perform: prepare_data, train, or evaluate"
    )
    args = parser.parse_args()
    main(args.action)
