from train.register import load_pickle, train_and_log_model

import os
import pickle
import tempfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import mlflow


def test_load_pickle():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
    data = {"key": "value"}
    with open(filename, "wb") as f_out:
        pickle.dump(data, f_out)

    loaded_data = load_pickle(filename)

    assert loaded_data == data

    os.remove(filename)
    
def test_train_and_log_model():
    x, y = np.random.rand(100, 10), np.random.rand(100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    params = {
        "max_depth": 2,
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run():
        for param in params:
            params[param] = int(params[param])

        random_forest = RandomForestRegressor(**params)
        random_forest.fit(x_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, random_forest.predict(x_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, random_forest.predict(x_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

    # Check if a new run has been created
    assert mlflow.active_run() is not None

    # End the run after the test
    mlflow.end_run()