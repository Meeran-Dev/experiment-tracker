from experiment_tracker.logger import ExperimentLogger

def test_new_logger_creates_empty_file(log_path):
    logger = ExperimentLogger(log_path)
    assert logger.get_history() == []


def test_log_run_returns_record(log_path):
    logger = ExperimentLogger(log_path)
    record = logger.log_run(accuracy=0.95, params={"max_iter": 200})
    assert record["accuracy"] == 0.95
    assert record["params"] == {"max_iter": 200}
    assert "timestamp" in record


def test_log_run_persists_to_history(log_path):
    logger = ExperimentLogger(log_path)
    logger.log_run(accuracy=0.9)
    logger.log_run(accuracy=0.93)

    history = logger.get_history()
    assert len(history) == 2
    assert history[0]["accuracy"] == 0.9
    assert history[1]["accuracy"] == 0.93


def test_history_survives_new_logger_instance(log_path):
    logger1 = ExperimentLogger(log_path)
    logger1.log_run(accuracy=0.88)

    # Simulate reopening the log in a new process/session
    logger2 = ExperimentLogger(log_path)
    history = logger2.get_history()
    assert len(history) == 1
    assert history[0]["accuracy"] == 0.88
