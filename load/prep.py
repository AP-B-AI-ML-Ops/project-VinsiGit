# pylint: disable=[invalid-name]

"""
This module contains tasks and flows for preprocessing earthquake data. 
It includes tasks for reading dataframes, preprocessing them 
and dumping the preprocessed data into pickle files. 
The main flow, `prep_flow`, orchestrates the execution of these tasks.
"""

import os
import pickle

import pandas as pd
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer


@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task
def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    return df


@task
def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    numerical = ["latitude", "longitude"]
    dicts = df[numerical].to_dict(orient="records")
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@flow
def prep_flow(data_path: str, dest_path: str):
    # Load parquet files
    df_train = read_dataframe(os.path.join(data_path, "earthquake-train.csv"))
    df_val = read_dataframe(os.path.join(data_path, "earthquake-val.csv"))
    df_test = read_dataframe(os.path.join(data_path, "earthquake-test.csv"))

    # Extract the target
    target = "mag"
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Keep only "latitude" and "longitude" for train data
    df_train = df_train[["latitude", "longitude"]]
    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
