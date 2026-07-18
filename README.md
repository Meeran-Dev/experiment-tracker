# experiment-tracker

A tiny ML experiment runner built while learning MLOps fundamentals

## What it does

- `Dataset` — loads a CSV and splits it into train/test sets
- `LogisticRegressionModel` — wraps a scikit-learn `LogisticRegression` with train/evaluate/predict
- `RandomForestModel` — wraps a scikit-learn `RandomForestClassifier` with train/evaluate/predict
- `ExperimentLogger` — logs each run's accuracy + params to a local JSON file
- `decorators` — `@timer`, `@log_errors`, `@retry(n)` used to instrument `Model.train`

## Install (editable / dev mode)

```bash
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
from experiment_tracker import Dataset, LogisticRegressionModel, ExperimentLogger

ds = Dataset("iris.csv")
ds.load()
X_train, X_test, y_train, y_test = ds.split(test_size=0.2)

model = LogisticRegressionModel(max_iter=200)
model.train(X_train, y_train)
acc = model.evaluate(X_test, y_test)

logger = ExperimentLogger("experiments.json")
logger.log_run(accuracy=acc, params={"max_iter": 200})
```

## Run tests

```bash
pytest -v
```

## Project structure

```
experiment-tracker/
├── pyproject.toml
├── src/experiment_tracker/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── logger.py
│   └── decorators.py
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_logger.py
│   └── test_decorators.py
└── README.md
```
