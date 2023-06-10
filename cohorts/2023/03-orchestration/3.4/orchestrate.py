import pathlib
import pickle
import polars as pl
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task
import os



@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pl.DataFrame:
    """Read data into DataFrame"""
    df = pl.read_parquet(filename)
    df = df.with_columns(
        (pl.col('lpep_dropoff_datetime') -
         pl.col('lpep_pickup_datetime')).alias('duration'))
    df = df.with_columns([
        (pl.col('duration').dt.seconds() / 60)])
    df = df.filter(
        pl.any((pl.col('duration') >= 1) &
               (pl.col('duration') <= 60)))
    categorical = ['PULocationID', 'DOLocationID']
    df = df.with_columns(pl.col(categorical).cast(pl.Utf8))

    return df


@task
def add_features(df_train: pl.DataFrame,
                 df_val: pl.DataFrame
        ) -> tuple[
        scipy.sparse.spmatrix,
        scipy.sparse.spmatrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer
    ]:
    """Add features to the model"""

    df_train = df_train.with_columns(pl.concat_str(
              ['PULocationID', 'DOLocationID'], separator="_"
            ).alias('PU_DO')
    )
    df_val = df_val.with_columns(pl.concat_str(
              ['PULocationID', 'DOLocationID'], separator="_"
            ).alias('PU_DO')
    )

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df_train.select(pl.col(categorical + numerical)).to_dicts()
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val.select(pl.col(categorical + numerical)).to_dicts()
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].to_numpy()
    y_val = df_val["duration"].to_numpy()

    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(X_train: scipy.sparse.spmatrix,
                     X_val: scipy.sparse.spmatrix,
                     y_train: np.ndarray,
                     y_val: np.ndarray,
                     dv: sklearn.feature_extraction.DictVectorizer,
            ) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    return None


@flow
def main_flow(
    train_path: str = "../data/green_tripdata_2021-01.parquet",
    val_path: str = "../data/green_tripdata_2021-02.parquet",
) -> None:
    """The main training pipeline"""

    print(os.getcwd())

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)


if __name__ == "__main__":
    main_flow()
