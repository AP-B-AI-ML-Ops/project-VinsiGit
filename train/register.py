# pylint: disable=[too-many-arguments,too-many-locals]

import os
import pickle

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def train_and_log_model(x_train, y_train, x_val, y_val, x_test, y_test, params):
    xgboost_params = [
        "max_depth",
        "n_estimators",
        "random_state",
        "n_jobs",
    ]

    with mlflow.start_run():
        for param in xgboost_params:
            params[param] = int(params[param])

        xgboost = XGBRegressor(**params)
        xgboost.fit(x_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, xgboost.predict(x_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, xgboost.predict(x_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


@task
def get_experiment_runs(top_n, hpo_experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(hpo_experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )
    return runs


@task
def select_best_model(top_n, experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    return best_run


@flow
def register_flow(
    path_to_model: str, top_n: int, experiment_name: str, hpo_experiment_name: str
):
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    x_train, y_train = load_pickle(os.path.join(path_to_model, "train.pkl"))
    x_val, y_val = load_pickle(os.path.join(path_to_model, "val.pkl"))
    x_test, y_test = load_pickle(os.path.join(path_to_model, "test.pkl"))

    # Retrieve the top_n model runs and log the models
    runs = get_experiment_runs(top_n, hpo_experiment_name)
    for run in runs:
        train_and_log_model(
            x_train, y_train, x_val, y_val, x_test, y_test, params=run.data.params
        )

    # Select the model with the lowest test RMSE
    best_run = select_best_model(top_n, experiment_name)

    best_params = best_run.data.params
    model = XGBRegressor(**best_params)

    # Fit your model
    model.fit(x_train, y_train)
    # model.fit(x_train, y_train)

    with mlflow.start_run() as run:
        # Log the model
        mlflow.xgboost.log_model(model, "model")

        # Get the run ID of the current run
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="xgboost-best-model")
