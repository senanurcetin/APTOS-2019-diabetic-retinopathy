"""Shared pytest configuration.

`tests/test_preprocessing.py` deliberately imports no pytest: it carries its own
collector so the preprocessing module can be checked anywhere Python and OpenCV
exist, including inside the Colab notebook that imports it straight from GitHub.
That property is worth keeping, so its tests are marked from here instead of
with a `pytestmark` line in the file itself.
"""
import pytest

# Test modules that need neither torch nor the dataset, but do not mark
# themselves. Anything added here must genuinely run on numpy and OpenCV alone -
# the fast CI job selects on this marker across three Python versions.
SELF_CONTAINED_MODULES = {"test_preprocessing"}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.module.__name__ in SELF_CONTAINED_MODULES:
            item.add_marker(pytest.mark.pure)
