import functools
import time

def timer(func):
    """Print how long the wrapped function took to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

def log_errors(func):
    """Catch exceptions from the wrapped function, log them, and re-raise."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[log_errors] {func.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper

def retry(n=3):
    """Retry the wrapped function up to n times on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"[retry] {func.__name__} attempt {attempt}/{n} failed: {e}")
            raise last_exception
        return wrapper
    return decorator