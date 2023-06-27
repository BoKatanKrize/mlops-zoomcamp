import sys
import pickle

import pandas as pd
import numpy as np


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


year = int(sys.argv[1])   # 2022
month = int(sys.argv[2])  # 3

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Mean predicted duration
print(f'Mean predicted duration: {np.mean(y_pred)}')

# Let's create an artificial 'ride_id' column:
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# Next, write the 'ride_id' and the predictions to a dataframe with results.
# (Save it as parquet):

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['prediction'] = y_pred

output_file = f'./yellow_{year:04d}-{month:02d}'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Let's now make the script configurable via CLI. We'll create two
# parameters: year and month.

# Run the script for March 2022

