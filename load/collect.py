import os
import urllib.parse
import urllib.request

import pandas as pd
from prefect import flow, task

DATALIMIT = 20000


@task
def generate_query_params(year, limit):
    params = {
        "starttime": f"{year}-01-01",
        "endtime": f"{year}-12-31",
        "minmagnitude": 2.5,
        "maxmagnitude": 5,
        "orderby": "time",
        "limit": limit,
    }
    return params


@task
def build_query_url(url, params):
    query = urllib.parse.urlencode(params)
    url = f"{url}?{query}"
    return url


@task(retries=4, retry_delay_seconds=2)
def load_data(url):
    data = pd.read_csv(url)
    return data


@task
def save_data(data, filename):
    data.to_csv(filename, index=False)


@flow
def collect_flow(data_path: str):
    os.makedirs(data_path, exist_ok=True)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"

    paramstest = generate_query_params(2021, DATALIMIT)
    paramsval = generate_query_params(2022, DATALIMIT)
    paramstrain = generate_query_params(2023, DATALIMIT)

    urltest = build_query_url(url, paramstest)
    urlval = build_query_url(url, paramsval)
    urltrain = build_query_url(url, paramstrain)

    datatest = load_data(urltest)
    dataval = load_data(urlval)
    datatrain = load_data(urltrain)

    save_data(datatest, os.path.join(data_path, "earthquake-test.csv"))
    save_data(dataval, os.path.join(data_path, "earthquake-val.csv"))
    save_data(datatrain, os.path.join(data_path, "earthquake-train.csv"))
