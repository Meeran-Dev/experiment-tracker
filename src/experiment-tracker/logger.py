import json
import os
from datetime import datetime, timezone


class ExperimentLogger:
    """Logs experiment runs (params + metrics) to a local JSON file."""

    def __init__(self, log_path: str = "experiments.json"):
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def log_run(self, accuracy: float, params: dict | None = None) -> dict:
        """Append a run record and return it."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": accuracy,
            "params": params or {},
        }
        history = self.get_history()
        history.append(record)
        with open(self.log_path, "w") as f:
            json.dump(history, f, indent=2)
        return record

    def get_history(self) -> list:
        """Return all logged runs."""
        with open(self.log_path, "r") as f:
            return json.load(f)
