# pylint: disable=[too-many-arguments,missing-module-docstring]

import os
import pickle

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def train_and_log_model(x_train, y_train, x_val, y_val, x_test, y_test, params):
    random_forest_params = [
        "max_depth",
        "n_estimators",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
        "n_jobs",
    ]

    with mlflow.start_run():
        for param in random_forest_params:
            params[param] = int(params[param])

        random_forest = RandomForestRegressor(**params)
        random_forest.fit(x_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, random_forest.predict(x_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, random_forest.predict(x_test), squared=False)
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
    model_path: str, top_n: int, experiment_name: str, hpo_experiment_name: str
):
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    x_train, y_train = load_pickle(os.path.join(model_path, "train.pkl"))
    x_val, y_val = load_pickle(os.path.join(model_path, "val.pkl"))
    x_test, y_test = load_pickle(os.path.join(model_path, "test.pkl"))

    # Retrieve the top_n model runs and log the models
    runs = get_experiment_runs(top_n, hpo_experiment_name)
    for run in runs:
        train_and_log_model(
            x_train, y_train, x_val, y_val, x_test, y_test, params=run.data.params
        )

    # Select the model with the lowest test RMSE
    best_run = select_best_model(top_n, experiment_name)

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="rf-best-model")
