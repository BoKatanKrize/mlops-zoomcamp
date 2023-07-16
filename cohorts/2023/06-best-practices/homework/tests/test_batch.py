from datetime import datetime
import pandas as pd
from pandas.testing import assert_frame_equal

# Pytest will find the first directory at or above the level of
# the file that does not include an __init__.py file (in this case
# homework/) and declare that directory the "basedir". It then adds
# the basedir to sys.path
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():

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
    categorical = ['PULocationID', 'DOLocationID']

    df = pd.DataFrame(data, columns=columns)
    df = prepare_data(df, categorical)

    expected_df = pd.DataFrame({'PULocationID': ['-1','1','1'],
                                'DOLocationID': ['-1','-1','2'],
                                'tpep_pickup_datetime': [dt(1, 2), dt(1, 2), dt(2, 2)],
                                'tpep_dropoff_datetime': [dt(1, 10), dt(1, 10), dt(2, 3)],
                                'duration': [8.0, 8.0, 1.0]})

    assert_frame_equal(df, expected_df, check_dtype=False)