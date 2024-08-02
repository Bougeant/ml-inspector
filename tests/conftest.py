""" Configuration for pytest. """

from .fixtures import (  # noqa: imported so that pytest can find the fixtures
    binary_predictions,
    multiclass_predictions,
)
