"""Fixtures for unit tests."""

import numpy as np
import pytest


@pytest.fixture
def binary_predictions():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob_1 = np.array([[0.7, 0.2], [0.1, 0.8], [0.4, 0.6], [0.0, 1.0], [0.7, 0.3]])
    y_prob_2 = np.array([[0.3, 0.7], [0.2, 0.7], [0.6, 0.4], [0.4, 0.6], [0.7, 0.3]])
    return y_true, y_prob_1, y_prob_2


@pytest.fixture
def multiclass_predictions():
    y_true = np.array([0, 1, 3, 1, 2, 0])
    y_prob_1 = np.array(
        [
            [0.5, 0.2, 0.1, 0.2],
            [0.1, 0.8, 0.0, 0.1],
            [0.4, 0.1, 0.2, 0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.2, 0.4, 0.3, 0.1],
            [0.7, 0.1, 0.1, 0.1],
        ]
    )
    y_prob_2 = np.array(
        [
            [0.3, 0.3, 0.2, 0.2],
            [0.2, 0.6, 0.1, 0.1],
            [0.4, 0.1, 0.2, 0.3],
            [0.2, 0.8, 0.0, 0.0],
            [0.3, 0.4, 0.2, 0.1],
            [0.6, 0.1, 0.1, 0.2],
        ]
    )
    return y_true, y_prob_1, y_prob_2
