"""experiment-tracker: a tiny ML experiment runner built for learning MLOps basics."""

from experiment_tracker.dataset import Dataset
from experiment_tracker.model import LogisticRegressionModel, RandomForestModel, BaseModel
from experiment_tracker.logger import ExperimentLogger
from experiment_tracker.decorator import timer, log_errors, retry

__all__ = [
    "Dataset",
    "LogisticRegressionModel",
    "RandomForestModel",
    "BaseModel",
    "ExperimentLogger",
    "timer",
    "log_errors",
    "retry",
]

__version__ = "0.1.0"
