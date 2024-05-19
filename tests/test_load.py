
import os
import pandas as pd
import numpy as np
import tempfile
import pickle
from sklearn.feature_extraction import DictVectorizer

from load.collect import generate_query_params, build_query_url, load_data, save_data
from load.prep import dump_pickle, read_dataframe, preprocess


def test_generate_query_params():
    actual = generate_query_params(2016, 2000)

    expected = {
        "starttime": "2016-01-01",
        "endtime": "2016-12-31",
        "minmagnitude": 2.5,
        "maxmagnitude": 5,
        "orderby": "time",
        "limit": 2000,
    }
    assert actual == expected


def test_build_query_url():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
    expected = {
        "starttime": "2016-01-01",
        "endtime": "2016-12-31",
        "minmagnitude": 2.5,
        "maxmagnitude": 5,
        "orderby": "time",
        "limit": 20000,
    }
    actual = build_query_url(url, expected)
    print(actual)
    expected = f"""{url}?starttime={2016}-01-01&endtime={2016}-12-31&minmagnitude=2.5&maxmagnitude=5&orderby=time&limit=20000"""
    assert actual == expected

def test_load_data_ask_teacher():
    pass
def test_save_data_ask_teacher():
    pass



def test_dump_pickle():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

    # Create some data to pickle
    data = {"key": "value"}

    # Use dump_pickle to write the data
    dump_pickle(data, filename)

    # Load the data back from the file
    with open(filename, "rb") as f_in:
        loaded_data = pickle.load(f_in)

    # Check if the loaded data matches the original data
    assert loaded_data == data

    # Clean up the temporary file
    os.remove(filename)

def test_read_dataframe():
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        filename = tmp.name

    # Write some data to the CSV file
    df = pd.DataFrame({
        "latitude": ["1.23", "4.56", "7.89"],
        "longitude": ["9.87", "6.54", "3.21"],
    })
    df.to_csv(filename, index=False)

    # Use read_dataframe to read the data back
    loaded_df = read_dataframe(filename)

    # Check if the loaded DataFrame matches the original DataFrame
    pd.testing.assert_frame_equal(loaded_df, df)

    # Clean up the temporary file
    os.remove(filename)

def test_preprocess_false():
    # Create a DataFrame
    df = pd.DataFrame({
        "latitude": ["1.23", "4.56", "7.89"],
        "longitude": ["9.87", "6.54", "3.21"],
    })
    
    
    dv = DictVectorizer()


    dv.fit(df.to_dict('records'))
    X, dv = preprocess(df, dv, fit_dv=False)
    assert X.shape == (3, 6)
    assert set(dv.feature_names_) == {"latitude=1.23", "latitude=4.56", "latitude=7.89", "longitude=9.87", "longitude=6.54", "longitude=3.21"}

def test_preprocess_true():
    df = pd.DataFrame({
        "latitude": ["1.23", "4.56", "7.89"],
        "longitude": ["9.87", "6.54", "3.21"],
    })

    dv = DictVectorizer()

    X, dv = preprocess(df, dv, fit_dv=True)
    
    
    assert X.shape == (3, 6)
