import pytest
import numpy as np


@pytest.fixture
def y_data():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def matching_y_pred():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def non_matching_y_pred():
    return np.array([1.1, 2.1, 3.1])
