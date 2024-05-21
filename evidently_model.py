# pylint: disable=[not-context-manager]

"""
This module contains functions for preparing and monitoring a database for an earthquake prediction model. 
It includes functions for preparing the database, preparing data, calculating metrics, and saving metrics to the database. 
The main function, `monitor`, orchestrates the execution of these functions.
"""


import os

import mlflow
import mlflow.pyfunc
import pandas as pd
import psycopg
from dotenv import load_dotenv
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from mlflow.tracking import MlflowClient

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
        mag float,
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
    ref_data = pd.read_csv("data/earthquake-val.csv")  # earthquake-
    raw_data = pd.read_csv("data/earthquake-train.csv")  # earthquake-

    client = MlflowClient()
    model_name = "xgboost-best-model"
    model_version = client.get_latest_versions(model_name, stages=["None"])[0]

    # model_uri = f"models:/{model_name}/{model_version.version}"
    model_uri = f"models:/{model_name}/{model_version.version}"

    model = mlflow.pyfunc.load_model(model_uri)

    print(ref_data.describe())
    val_preds = model.predict(ref_data[NUMERICAL])
    ref_data["prediction"] = val_preds

    val_preds = model.predict(raw_data[NUMERICAL])
    raw_data["prediction"] = val_preds

    # ref_columns_without_prediction = ref_data.columns.drop("prediction")
    # raw_data = raw_data[ref_columns_without_prediction]

    return ref_data, model, raw_data


def calculate_metrics(ref_data, model, current_data):
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
        mag,
        prediction_drift,
        num_drifted_columns,
        share_missing_values
    )
    VALUES (%s, %s, %s, %s);
    """,
        (date, prediction_drift, num_drifted_cols, share_missing_vals),
    )


def monitor():
    mlflow.set_tracking_uri("")
    startMag = 2.5
    endMag = 2.6

    prep_db()

    ref_data, model, raw_data = prep_data()

    with psycopg.connect(f"{CONNECT_STRING} dbname=test") as conn:
        with conn.cursor() as cursor:
            for i in range(0, 24):
                current_data = raw_data[
                    (raw_data.mag >= startMag) & (raw_data.mag < endMag)
                ]
                (
                    prediction_drift,
                    num_drifted_cols,
                    share_missing_vals,
                ) = calculate_metrics(ref_data, model, current_data)
                save_metrics_to_db(
                    cursor,
                    startMag,
                    prediction_drift,
                    num_drifted_cols,
                    share_missing_vals,
                )

                startMag += 0.1
                endMag += 0.1

                # time.sleep(1)
                print(i)


if __name__ == "__main__":
    monitor()
