# pylint: disable=[missing-module-docstring]

import os
import pickle

import mlflow
from prefect import flow, task
from xgboost import XGBRegressor


@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def start_ml_experiment(x_train, y_train):
    with mlflow.start_run():
        xgb_regressor = XGBRegressor(max_depth=10, random_state=0)
        xgb_regressor.fit(x_train, y_train)


@flow
def train_flow(path_to_model: str):
    mlflow.set_experiment("xgboost-train")
    mlflow.sklearn.autolog()

    x_train, y_train = load_pickle(os.path.join(path_to_model, "train.pkl"))

    start_ml_experiment(x_train, y_train)
