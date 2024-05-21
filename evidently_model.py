# pylint: disable=[not-context-manager]

"""
This module contains functions for preparing and monitoring a database for an earthquake prediction model. 
It includes functions for preparing the database, preparing data, calculating metrics, and saving metrics to the database. 
The main function, `monitor`, orchestrates the execution of these functions.
"""


import datetime
import os

import pickle
import pandas as pd
import psycopg
from dotenv import load_dotenv
import mlflow
from train.register import select_best_model

from evidently import ColumnMapping
from evidently.metrics import (ColumnDriftMetric, DatasetDriftMetric,
                               DatasetMissingValuesMetric)
from evidently.report import Report

load_dotenv()
NUMERICAL = ["latitude", "longitude"]

CATEGORICAL = []

COL_MAPPING = ColumnMapping(
    prediction="prediction",
    numerical_features=NUMERICAL,
    categorical_features=CATEGORICAL,
    target="mag",  # None,
)

# host, port, user, password
CONNECT_STRING = f"""
host={os.getenv("POSTGRES_HOST")}
port={os.getenv("POSTGRES_PORT")}
user={os.getenv("POSTGRES_USER")}
password={os.getenv("POSTGRES_PASSWORD")}"""


def prep_db():
    create_table_query = """
    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics(
        timestamp timestamp,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float
    );
    """

    with psycopg.connect(CONNECT_STRING, autocommit=True) as conn:
        # zoek naar database genaamd 'test' in de metadata van postgres
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect(f"{CONNECT_STRING} dbname=test") as conn:
            conn.execute(create_table_query)


    
def prep_data():
    # ref_data = pd.read_parquet("data/val.pkl")
    with open(os.path.join("models", "val.pkl"), 'rb') as f:
        ref_data = pickle.load(f)

    best_model = select_best_model(top_n=1, experiment_name="xgboost-best-model")
    run_id = best_model.info.run_id
    model_uri = f"runs:/{run_id}/model"

    # model_uri = "models:/xgboost-best-model/Production"
    model = mlflow.pyfunc.load_model(model_uri, name="xgboost-best-model")

    raw_data = pd.read_csv("data/earthquake-train.csv")  # earthquake-
    ref_columns_without_prediction = ref_data.columns.drop("prediction")
    raw_data = raw_data[ref_columns_without_prediction]

    return ref_data, model, raw_data


def calculate_metrics(current_data, model, ref_data):
    features = [
        feature
        for feature in NUMERICAL + CATEGORICAL
        if feature in current_data.columns
    ]

    current_data["prediction"] = model.predict(current_data[features].fillna(0))

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    report.run(
        reference_data=ref_data, current_data=current_data, column_mapping=COL_MAPPING
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_cols = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_vals = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]

    return prediction_drift, num_drifted_cols, share_missing_vals


def save_metrics_to_db(
    cursor, date, prediction_drift, num_drifted_cols, share_missing_vals
):
    cursor.execute(
        """
    INSERT INTO metrics(
        timestamp,
        prediction_drift,
        num_drifted_columns,
        share_missing_values
    )
    VALUES (%s, %s, %s, %s);
    """,
        (date, prediction_drift, num_drifted_cols, share_missing_vals),
    )


def monitor():
    start_date = datetime.datetime(2023, 1, 1, 0, 0)
    end_date = datetime.datetime(2023, 6, 1, 0, 0)

    prep_db()

    ref_data, model, raw_data = prep_data()

    with psycopg.connect(f"{CONNECT_STRING} dbname=test") as conn:
        with conn.cursor() as cursor:
            for i in range(0, 30):
                current_data = raw_data
                (
                    prediction_drift,
                    num_drifted_cols,
                    share_missing_vals,
                ) = calculate_metrics(current_data, model, ref_data)
                save_metrics_to_db(
                    cursor,
                    start_date,
                    prediction_drift,
                    num_drifted_cols,
                    share_missing_vals,
                )

                start_date += datetime.timedelta(1)
                end_date += datetime.timedelta(1)

                # time.sleep(1)
                print(i)


if __name__ == "__main__":
    monitor()
