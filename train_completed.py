import pickle
import pandas as pd
import numpy as np

from boxkite.monitoring.service import ModelMonitoringService
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def main():
    #now add a drift
    dataset_red = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        delimiter=";", header=0)
    dataset_white = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        delimiter=";", header=0)
    dataset = pd.concat([dataset_red, dataset_white], axis=0)
    conditions = [
        (dataset["quality"] <= 3),
        (dataset.quality > 3) & (dataset.quality <= 6),
        (dataset["quality"] > 6)
    ]
    values = [0, 1, 2]
    dataset['quality'] = np.select(conditions, values)
    target = dataset["quality"]
    covariates = dataset.loc[:, dataset.columns != "quality"]
    X_train, X_test, Y_train, Y_test = train_test_split(covariates, target)
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_pred = np.round(Y_pred)
    print("Score: %.2f" % r2_score(Y_test, Y_pred))
    with open("./model.pkl", "wb") as f:
        pickle.dump(model, f)

    features = zip(*[covariates.columns, X_train.apply(tuple, axis=1)])
#    features = zip(*[bunch.feature_names, X_train.T])
    # features = [("age", [33, 23, 54, ...]), ("sex", [0, 1, 0]), ...]
    inference = list(Y_pred)
    # if raises an error: remove this
    inference = np.round(inference,2)
    # inference = [235.01351432, 211.79644624, 121.54947698, ...]
    ModelMonitoringService.export_text(
        features=features,
        inference=inference,
        path="./histogram.prom",
    )


if __name__ == "__main__":
    main()

