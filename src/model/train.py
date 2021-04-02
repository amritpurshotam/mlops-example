import json

import click
import pandas as pd
import sklearn.metrics as metrics
import yaml
from joblib import dump
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.dataset import load_dataset
from src.feature.preprocess import create_pipeline


def create_model() -> BaseEstimator:
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    model = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        n_jobs=-1,
    )
    return model


def evaluate(pipeline: Pipeline, features: DataFrame, labels: DataFrame) -> None:
    predictions = pipeline.predict(features)
    bal_acc = metrics.balanced_accuracy_score(labels, predictions)
    with open("scores.json", "w") as fd:
        json.dump({"bal_acc": bal_acc}, fd, indent=4)

    actual = pd.DataFrame(labels.values, columns=["actual"])
    predictions = pd.DataFrame(predictions, columns=["predictions"])
    pd.concat([actual, predictions], axis=1).to_csv("cm.csv", index=False)


def train(features: DataFrame, labels: DataFrame) -> Pipeline:
    model = create_model()
    pipeline = create_pipeline(model)
    pipeline.fit(features, labels)
    return pipeline


@click.command()
@click.option("--data_path", type=str, required=True, help="Path to the training data.")
@click.option(
    "--model_output_path",
    type=str,
    required=True,
    help="Path to save the trained model.",
)
def run(data_path: str, model_output_path: str):
    train_features, val_features, _, train_labels, val_labels, _ = load_dataset(
        data_path
    )
    pipeline = train(train_features, train_labels)
    evaluate(pipeline, val_features, val_labels)
    dump(pipeline, model_output_path)


if __name__ == "__main__":
    run()
