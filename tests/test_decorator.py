import pytest

from experiment_tracker.decorator import timer, log_errors, retry


def test_timer_returns_original_result(capsys):
    @timer
    def add(a, b):
        return a + b

    result = add(2, 3)
    assert result == 5
    captured = capsys.readouterr()
    assert "took" in captured.out


def test_log_errors_reraises_and_logs(capsys):
    @log_errors
    def boom():
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        boom()
    captured = capsys.readouterr()
    assert "ValueError" in captured.out


def test_retry_succeeds_after_failures():
    attempts = {"count": 0}

    @retry(n=3)
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("not yet")
        return "success"

    assert flaky() == "success"
    assert attempts["count"] == 3


def test_retry_raises_after_exhausting_attempts():
    @retry(n=2)
    def always_fails():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        always_fails()
