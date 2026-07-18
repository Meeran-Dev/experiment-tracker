import pytest

from experiment_tracker.dataset import Dataset


def test_load_returns_dataframe(iris_csv):
    ds = Dataset(iris_csv)
    df = ds.load()
    assert len(df) == 150
    assert df.shape[1] == 5  # 4 features + target


def test_split_before_load_raises(iris_csv):
    ds = Dataset(iris_csv)
    with pytest.raises(ValueError):
        ds.split()


@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.5])
def test_split_proportions(iris_csv, test_size):
    ds = Dataset(iris_csv)
    ds.load()
    X_train, X_test, y_train, y_test = ds.split(test_size=test_size)

    total = len(X_train) + len(X_test)
    assert total == 150
    assert abs(len(X_test) / total - test_size) < 0.02
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
