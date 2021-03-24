import json

import click
import numpy as np
import yaml
from joblib import dump
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def evaluate(features: DataFrame, labels: DataFrame) -> None:
    model = create_model()
    pipeline = create_pipeline(model)

    cv = StratifiedKFold(n_splits=5, random_state=42)
    scores = cross_val_score(
        pipeline, features, labels, cv=cv, scoring="balanced_accuracy", n_jobs=-1
    )

    bal_acc = np.average(scores)
    with open("scores.json", "w") as fd:
        json.dump({"bal_acc": bal_acc}, fd, indent=4)


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
    train_features, _, train_labels, _ = load_dataset(data_path)
    evaluate(train_features, train_labels)
    pipeline = train(train_features, train_labels)
    dump(pipeline, model_output_path)


if __name__ == "__main__":
    run()
