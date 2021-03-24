import click
import yaml
from joblib import dump
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.dataset import load_dataset
from src.feature.preprocess import create_pipeline


def train(features: DataFrame, labels: DataFrame) -> Pipeline:
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    model = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        n_jobs=-1,
    )
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
    pipeline = train(train_features, train_labels)
    dump(pipeline, model_output_path)


if __name__ == "__main__":
    run()
