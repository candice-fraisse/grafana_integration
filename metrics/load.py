from argparse import ArgumentParser
from random import gauss
from time import sleep

import numpy as np
import pandas as pd
from requests import Session

from provisioning import drifts


def import_dataset() -> pd.DataFrame:
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
    return dataset


def corrupt_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset_generator = drifts.dataset_generator_yield(data=dataset, nb_sample=100)
    dataset_sampled = next(dataset_generator)
    #dataset_corrupted = drifts.drift_generator_concept_drift(data=dataset_sampled,
    #                                                         column_name="alcohol",
    #                                                         value=11,
    #                                                         label_col="quality",
    #                                                         label_value=0,
    #                                                         action="greater")
    dataset_corrupted = drifts.drift_generator_univariate_increase(data=dataset_sampled,
                                                                   column_name="alcohol",
                                                                   value=100)
    dataset_sampled_not_corrupted = next(dataset_generator)
    dataset = pd.concat([dataset_sampled_not_corrupted, dataset_corrupted], axis=0)
    X_corrupted = dataset.loc[:,dataset.columns != "quality"]
    return X_corrupted

def main(classification):
    dataset = import_dataset()
    X_corrupted = corrupt_dataset(dataset)
    data = X_corrupted.values.tolist()
    with Session() as s:
        for i in range(200):
        #    data[0] = gauss(6, 2) if classification else gauss(0, 0.05)
            resp = s.post("http://localhost:5000", json=data[i])
            resp.raise_for_status()
            if i % 10 == 0:
                print(f"response[{i}]: {resp.json()}")
            sleep(0.1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate inference workload.")
    parser.add_argument("-c", "--classification", action="store_true")
    args = parser.parse_args()
    main(args.classification)
