import os
import pickle
import click
import mlflow
# An open source hyperparameter optimization framework
# to automate hyperparameter search
import optuna
from typing import Any

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str) -> Any:
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int) -> None:

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # 1. Define an objective function to be minimized
    def objective(trial):

        with mlflow.start_run():
            # 2. Suggest values for the hyperparameters using a trial object
            params = {
                # Number of trees in the forest.
                'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                # Minimum number of samples required to split an internal node
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
                # Minimum number of samples required to be at a leaf node
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
                'random_state': 42,
                'n_jobs': -1
            }
            mlflow.set_tag("model", "RandomForestRegressor")
            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

            return rmse

    sampler = TPESampler(seed=42) # Tree of Parzen Estimators (TPE) algorithm
    # 3. Create a study object and optimize the objective function.
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    run_optimization()
