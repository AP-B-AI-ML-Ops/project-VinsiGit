import os
import pickle
import tempfile

from train.register import load_pickle


def test_load_pickle():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
    data = {"key": "value"}
    with open(filename, "wb") as f_out:
        pickle.dump(data, f_out)

    loaded_data = load_pickle(filename)

    assert loaded_data == data

    os.remove(filename)
