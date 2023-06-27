import polars as pl

import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import sys
import uuid


# Create unique IDs for the DataFrame rows
def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


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
    # Add unique IDs as new column
    series_uuids = pl.Series(name="ride_id", 
                             values=generate_uuids(len(df)))
    df = df.with_columns(series_uuids)
    return df


def prepare_dictionaries(df: pl.DataFrame) ->  dict[str,float|str]:
    df= df.with_columns(pl.concat_str(
                        ['PULocationID','DOLocationID'],
                        separator="_").alias('PU_DO'))
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df.select(pl.col(categorical+numerical)).to_dicts()
    return dicts


# Load the model obtained in 4.3
def load_model(run_id):
    logged_model = f'runs:/{run_id}/model'            
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    print(f'\nreading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'\nloading the model with RUN_ID={run_id}...')
    model = load_model(run_id)

    print(f'\napplying the model...')
    y_pred = model.predict(dicts)

    print(f'\nsaving the result to {output_file}...')
    df_result = pl.DataFrame()
    df_result = df_result.with_columns(df['ride_id'])
    df_result = df_result.with_columns(df['lpep_pickup_datetime'])
    df_result = df_result.with_columns(df['PULocationID'])
    df_result = df_result.with_columns(df['DOLocationID'])
    # Column for the duration given in input file 
    df_result = df_result.with_columns(df['duration'].alias('actual_duration'))
    # Column for predicted duration
    y_pred = pl.Series(name='predicted_duration', values=y_pred)
    df_result = df_result.with_columns(y_pred)
    # Column with difference between actual and prediction
    df_result = df_result.with_columns(
                (pl.col('actual_duration') -
                 pl.col('predicted_duration')).alias('diff'))
    # Same model for every row
    df_result = df_result.with_columns(
        pl.lit(run_id, dtype=pl.Utf8).alias("model_version")
    )
    # Write output to parquet file
    df_result.write_parquet(output_file)


def run():
    # We can parametrise our input and output files.
    # The output file stores the 'actual_duration' vs 'predicted_duration'
    taxi_type = sys.argv[1]   # 'green'
    year = int(sys.argv[2])   # 2021
    month = int(sys.argv[3])  # 2
    run_id = sys.argv[4]      # '068bda10a3ed4b73a771df771161f60a'

    input_file  = f'./data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'./output/{taxi_type}_{year:04d}-{month:02d}.parquet'

    apply_model(input_file=input_file,
                run_id=run_id,
                output_file=output_file)


if __name__ == "__main__":
    run()


