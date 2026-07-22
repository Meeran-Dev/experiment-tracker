FROM python:3.11-slim

COPY pyproject.toml .
COPY src/ src/
COPY run_experiment.py .
RUN pip install -e .
CMD ["python3", "run_experiment.py"]