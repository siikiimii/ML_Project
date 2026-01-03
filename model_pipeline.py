"""
Model pipeline functions for churn prediction.

This module provides functions to prepare data, train a model,
evaluate performance, and handle model persistence.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_data(path="data/Churn_Modelling.csv"):
    """
    Load and split the dataset into training and test sets.
    Drops identifier columns that are not useful for prediction.

    Args:
        path (str): Path to the dataset CSV file.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please copy it into the project directory."
        )

    df = pd.read_csv(path)

    # 🔑 Drop irrelevant identifier columns
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    # Features and target
    x_data = df.drop("Exited", axis=1)
    y_data = df["Exited"]

    return train_test_split(x_data, y_data, test_size=0.2, random_state=42)


def train_model(x_train, y_train):
    """
    Train a RandomForest model with preprocessing for categorical features.

    Args:
        x_train (DataFrame): Training features.
        y_train (Series): Training labels.

    Returns:
        Pipeline: Trained model pipeline.
    """
    categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = x_train.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on test data.

    Args:
        model (Pipeline): Trained model pipeline.
        x_test (DataFrame): Test features.
        y_test (Series): Test labels.

    Returns:
        float: Accuracy score.
    """
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def save_model(model, path="model.joblib"):
    """
    Save the trained model pipeline to disk.

    Args:
        model (Pipeline): Trained model pipeline.
        path (str): File path to save the model.
    """
    joblib.dump(model, path)


def load_model(path="model.joblib"):
    """
    Load a trained model pipeline from disk.

    Args:
        path (str): File path to load the model from.

    Returns:
        Pipeline: Loaded model pipeline.
    """
    return joblib.load(path)
