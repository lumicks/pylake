import pytest
from ..data.generate_exponential_data import read_dataset as read_dataset_exponential


@pytest.fixture
def exponential_data():
    return read_dataset_exponential("exponential_data.npz")
