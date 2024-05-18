from load import collect


def test_generate_query_params():
    actual = collect.generate_query_params(2016, 2000)

    expected = {
        "starttime": "2016-01-01",
        "endtime": "2016-12-31",
        "minmagnitude": 2.5,
        "maxmagnitude": 5,
        "orderby": "time",
        "limit": 2000,
    }
    assert actual == expected


def test_build_query_url():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
    expected = {
        "starttime": "2016-01-01",
        "endtime": "2016-12-31",
        "minmagnitude": 2.5,
        "maxmagnitude": 5,
        "orderby": "time",
        "limit": 2000,
    }
    actual = collect.build_query_url(url, expected)

    expected = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
    assert actual == expected
