import os
import pickle
import mlflow
import optuna

from prefect import task, flow

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def optimize(X_train, y_train, X_val, y_val, num_trials):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
            'random_state': 42,
            'n_jobs': -1
        }
        with mlflow.start_run():
            mlflow.log_params(params)
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


@flow
def hpo_flow(model_path: str, num_trials: int, experiment_name: str):
    mlflow.set_experiment(experiment_name)

    mlflow.sklearn.autolog(disable=True)

    X_train, y_train = load_pickle(os.path.join(model_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(model_path, "val.pkl"))

    optimize(X_train, y_train, X_val, y_val, num_trials)