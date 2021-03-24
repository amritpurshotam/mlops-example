import os
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def _read_in_raw_data(data_dir: str) -> Tuple[DataFrame, DataFrame]:
    """Read in the raw water pump features and labels.

    Parameters
    ----------
    data_dir : str
        Path of the directory where `water_pump_features.csv` and
        `water_pump_labels.csv` can be found.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        DataFrames of the features and labels respectively.
    """
    features_file_name = os.path.join(data_dir, "water_pump_features.csv")
    features = pd.read_csv(
        features_file_name,
        parse_dates=["date_recorded"],
        dtype={"region_code": str, "district_code": str},
    )

    labels_file_name = os.path.join(data_dir, "water_pump_labels.csv")
    labels = pd.read_csv(labels_file_name)

    return features, labels


def _align_features_and_labels(
    features: DataFrame, labels: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """Align the `feature`s and `labels` DataFrames so they're both in the same order
    removing the need to check the `id` columns in each.

    Parameters
    ----------
    features : DataFrame
        DataFrame containing the `id` attribute.
    labels : DataFrame
        DataFrame containing the `id` attribute that corresponds to the `id` in
        `features`.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        DataFrames of the features and labels respectively with the `id` column
        in `labels` dropped making it more amenable to pass into classifiers for
        training.
    """
    aligned = features.merge(labels, on="id", validate="one_to_one")

    labels = aligned["status_group"]
    features = aligned.drop(columns=["status_group"])

    return features, labels


def _split(
    features: DataFrame, labels: DataFrame
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Deterministic random 90/10 train/test split of the dataset stratified by the
    labels.

    Parameters
    ----------
    features : DataFrame

    labels : DataFrame

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame]
        A tuple of four DataFrames corresponding to the training features,
        testing features, training labels, and testing labels respectively.
    """
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.1, random_state=42, stratify=labels
    )
    return train_features, test_features, train_labels, test_labels


def load_dataset(data_dir: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Read in the water pump dataset and split into the training and test sets.

    Parameters
    ----------
    data_dir : str
        Path of the directory where `water_pump_features.csv` and
        `water_pump_labels.csv` can be found.

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame]
        A tuple of four DataFrames corresponding to the training features,
        testing features, training labels, and testing labels respectively.
    """
    features, labels = _read_in_raw_data(data_dir)
    features, labels = _align_features_and_labels(features, labels)
    train_features, test_features, train_labels, test_labels = _split(features, labels)

    return train_features, test_features, train_labels, test_labels
