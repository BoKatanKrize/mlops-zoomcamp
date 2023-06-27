import pickle

import pandas as pd
import polars as pl

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import mlflow


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


def prepare_dictionaries(df: pl.DataFrame) ->  dict[str,float|str]:
    df= df.with_columns(pl.concat_str(
                        ['PULocationID','DOLocationID'],
                        separator="_").alias('PU_DO'))
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df.select(pl.col(categorical+numerical)).to_dicts()
    return dicts


if __name__ == '__main__':

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("green-taxi-duration")

    df_train = read_dataframe('./data/green_tripdata_2021-01.parquet')
    df_val = read_dataframe('./data/green_tripdata_2021-02.parquet')

    target = 'duration'
    y_train = df_train[target].to_numpy()
    y_val = df_val[target].to_numpy()

    dict_train = prepare_dictionaries(df_train)
    dict_val = prepare_dictionaries(df_val)

    with mlflow.start_run():
        params = dict(max_depth=20,
                      n_estimators=100,
                      min_samples_leaf=10,
                      random_state=0)
        mlflow.log_params(params)

        pipeline = make_pipeline(
            DictVectorizer(),
            RandomForestRegressor(**params, n_jobs=-1)
        )

        pipeline.fit(dict_train, y_train)
        y_pred = pipeline.predict(dict_val)

        rmse = mean_squared_error(y_pred, y_val, squared=False)
        print(params, rmse)
        mlflow.log_metric('rmse', rmse)

        # The pipeline is logged as artifact (DictVectorizer + Random Forest)
        # instead of separately
        mlflow.sklearn.log_model(pipeline, artifact_path="model")