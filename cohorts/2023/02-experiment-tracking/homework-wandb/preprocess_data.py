import os
import pickle
import click
from typing import Any, Tuple

import polars as pl
import numpy as np

import wandb

from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj: Any, filename: str) -> None:
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)


def read_dataframe(filename: str) -> pl.DataFrame:
    df = pl.read_parquet(filename)
    df = df.with_columns(
                (pl.col('lpep_dropoff_datetime') -
                 pl.col('lpep_pickup_datetime')).alias('duration'))
    df = df.with_columns([
           (pl.col('duration').dt.seconds() / 60)])
    df = df.filter(
            pl.any((pl.col('duration') >= 1) &
                   (pl.col('duration') <= 60)))
    categorical = ['PULocationID','DOLocationID']
    df = df.with_columns(pl.col(categorical).cast(pl.Utf8))
    return df


def preprocess(df: pl.DataFrame,
               dv: DictVectorizer,
               fit_dv: bool = False) ->  Tuple[np.ndarray, DictVectorizer]:
    df= df.with_columns(pl.concat_str(
                        ['PULocationID','DOLocationID'],
                        separator="_").alias('PU_DO'))
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df.select(pl.col(categorical+numerical)).to_dicts()
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option(
    "--raw_data_path", help="Location where the raw NYC taxi trip data was saved"
)
@click.option("--dest_path", help="Location where the resulting files will be saved")
def run_data_prep(
    wandb_project: str,
    wandb_entity: str,
    raw_data_path: str,
    dest_path: str,
    dataset: str = "green",
) -> None:
    # Initialize a Weights & Biases run (sync and log data as a W&B Run)
    wandb.init(project=wandb_project, entity=wandb_entity, job_type="preprocess")

    # Load parquet files
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-01.parquet")
    )
    df_val = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-02.parquet")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-03.parquet")
    )

    # Extract the target
    target = "tip_amount"
    y_train = df_train[target].to_numpy()
    y_val   = df_val[target].to_numpy()
    y_test  = df_test[target].to_numpy()

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

    # Log your dataset as a versioned file to Weights & Biases Artifact
    artifact = wandb.Artifact("NYC-Taxi", type="preprocessed_dataset")
    artifact.add_dir(dest_path)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    run_data_prep()
