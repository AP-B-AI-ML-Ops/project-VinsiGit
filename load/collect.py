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

    params2022 = generate_query_params(2022, DATALIMIT)
    params2023 = generate_query_params(2023, DATALIMIT)
    params2024 = generate_query_params(2024, DATALIMIT)

    url2022 = build_query_url(url, params2022)
    url2023 = build_query_url(url, params2023)
    url2024 = build_query_url(url, params2024)

    data2022 = load_data(url2022)
    data2023 = load_data(url2023)
    data2024 = load_data(url2024)

    save_data(data2022, os.path.join(data_path, "earthquake-2022.csv"))
    save_data(data2023, os.path.join(data_path, "earthquake-2023.csv"))
    save_data(data2024, os.path.join(data_path, "earthquake-2024.csv"))
