from pathlib import Path

import numpy as np
import pytest

from .data.generate_exponential_data import read_dataset as read_dataset_exponential


def extract_param(data, n_states):
    keys = ("initial_state_prob", "transition_prob", "means", "st_devs")
    param = {"n_states": n_states}
    for key in keys:
        param[key] = data[f"{key}_{n_states}"]
    return param


@pytest.fixture(scope="session", params=[1, 2, 3, 4])
def trace_lownoise(request):
    """Trace data can be generated by running ./data/generate_trace_data.py"""

    data = np.load(Path(__file__).parent / "data/trace_data.npz")
    n_states = request.param

    param = extract_param(data, n_states)
    y = data[f"y_{n_states}"]
    sp = data[f"sp_{n_states}"]

    return y, sp, param


@pytest.fixture(scope="session")
def trace_simple(request):
    """Trace data can be generated by running ./data/generate_trace_data.py"""

    data = np.load(Path(__file__).parent / "data/trace_data.npz")
    n_states = 2

    param = extract_param(data, n_states)
    y = data[f"y_{n_states}"]
    sp = data[f"sp_{n_states}"]

    return y, sp, param


@pytest.fixture
def exponential_data():
    return read_dataset_exponential("exponential_data.npz")
