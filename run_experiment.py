from experiment_tracker import Dataset, LogisticRegressionModel, ExperimentLogger

dataset = Dataset("temp_data.csv")
model = LogisticRegressionModel(max_iter = 1000)
logger = ExperimentLogger("temp_experiments.json")

dataset.load()
X_train, X_test, y_train, y_test = dataset.split()

model.train(X_train, y_train)
accuracy = model.evaluate(X_test, y_test)
logger.log_run(accuracy, model.model_params)
