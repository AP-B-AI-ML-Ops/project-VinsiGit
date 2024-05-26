import os
import pickle

import mlflow
import optuna
from optuna.samplers import TPESampler
from prefect import flow, task
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def optimize(x_train, y_train, x_val, y_val, num_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "random_state": 42,
            "n_jobs": -1,
        }
        with mlflow.start_run():
            mlflow.log_params(params)
            xgboost = XGBRegressor(**params)
            xgboost.fit(x_train, y_train)
            y_pred = xgboost.predict(x_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


@flow
def hpo_flow(path_to_model: str, num_trials: int, experiment_name: str):
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(disable=True)

    x_train, y_train = load_pickle(os.path.join(path_to_model, "train.pkl"))
    x_val, y_val = load_pickle(os.path.join(path_to_model, "val.pkl"))

    optimize(x_train, y_train, x_val, y_val, num_trials)
