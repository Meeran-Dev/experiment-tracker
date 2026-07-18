import pandas as pd
import pytest
from sklearn.datasets import load_iris


@pytest.fixture
def iris_csv(tmp_path):
    """Write the iris dataset to a temp CSV and return its path.

    Using tmp_path (a built-in pytest fixture) means every test gets
    a fresh, isolated file -- no leftover state between tests.
    """
    iris = load_iris(as_frame=True)
    df = iris.frame  # includes feature columns + 'target' column
    csv_path = tmp_path / "iris.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def log_path(tmp_path):
    """Return a temp path for an ExperimentLogger's JSON file."""
    return str(tmp_path / "experiments.json")
