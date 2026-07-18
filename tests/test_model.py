import pytest

from experiment_tracker.dataset import Dataset
from experiment_tracker.model import LogisticRegression as Model


@pytest.fixture
def trained_model_and_split(iris_csv):
    ds = Dataset(iris_csv)
    ds.load()
    X_train, X_test, y_train, y_test = ds.split(test_size=0.2)
    model = Model(max_iter=200)
    model.train(X_train, y_train)
    return model, X_test, y_test


def test_predict_before_train_raises(iris_csv):
    ds = Dataset(iris_csv)
    ds.load()
    _, X_test, _, _ = ds.split()
    model = Model(max_iter=200)
    with pytest.raises(RuntimeError):
        model.predict(X_test)


def test_train_sets_is_trained_flag(trained_model_and_split):
    model, _, _ = trained_model_and_split
    assert model.is_trained is True


def test_evaluate_returns_valid_accuracy(trained_model_and_split):
    model, X_test, y_test = trained_model_and_split
    acc = model.evaluate(X_test, y_test)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_predict_returns_correct_length(trained_model_and_split):
    model, X_test, y_test = trained_model_and_split
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
