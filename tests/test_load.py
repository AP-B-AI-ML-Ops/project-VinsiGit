from load import collect

def test_generate_query_params():

    actual = collect.generate_query_params(2016,2000)

    expected = {
        "starttime": f"2016-01-01",
        "endtime": f"2016-12-31",
        "minmagnitude": 2.5,
        "maxmagnitude": 5,
        "orderby": "time",
        "limit": 2000
    }
    assert actual == expected