import os
from datetime import datetime
import pandas as pd
from pandas.testing import assert_frame_equal


os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

columns = ['PULocationID', 'DOLocationID',
           'tpep_pickup_datetime', 'tpep_dropoff_datetime']

df_input = pd.DataFrame(data, columns=columns)


# Save dataframe to LocalStack S3 bucket

input_file = "s3://nyc-duration/in/2022-01.parquet"

options = {
    'client_kwargs': {
        'endpoint_url': os.environ['S3_ENDPOINT_URL']
    }
}

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

# To see the input .parquet file in the S3 bucket, execute in console:
# aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in/

# Execute the batch.py script
year = 2022
month = 1
command = f'python batch.py {year} {month}'
os.system(command)

# To see the output .parquet file in the S3 bucket, execute in console:
# aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/out/

# Read predicted data saved in S3
output_file = "s3://nyc-duration/out/2022-01.parquet"
df_out = pd.read_parquet(output_file, storage_options=options)

# Define expected dataframe
expected_df_out = pd.DataFrame({'ride_id': ['2022/01_0',
                                            '2022/01_1',
                                            '2022/01_2'],
                                'predicted_duration': [24.781802,
                                                        0.617543,
                                                        6.108105]})
# verify the prediction saved in S3 is correct
assert_frame_equal(df_out, expected_df_out, check_dtype=False)

print('Integration test successfully completed')