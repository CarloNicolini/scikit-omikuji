import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, sparray, spmatrix
from skomikuji import OmikujiClassifier
from skomikuji.metrics import compute_metrics
from tabulate import tabulate
from tqdm.rich import tqdm
import logging

slices = [
    "X_test",
    "Y_test",
    "Y_test_pred",
    "X_train",
    "Y_train",
    "Y_train_pred",
    "Y_train_proba_pred",
    "Y_test_proba_pred",
]


def filter_no_labels(data):
    rows_ok_train = data["Y_train"].sum(1) > 0
    rows_ok_test = data["Y_test"].sum(1) > 0
    return {
        "X_train": data.pop("X_train")[rows_ok_train],
        "Y_train": data.pop("Y_train")[rows_ok_train],
        "Y_train_pred": data.pop("Y_train_pred")[rows_ok_train],
        "Y_train_proba_pred": data.pop("Y_train_proba_pred")[rows_ok_train],
        "X_test": data.pop("X_test")[rows_ok_test],
        "Y_test": data.pop("Y_test")[rows_ok_test],
        "Y_test_pred": data.pop("Y_test_pred")[rows_ok_test],
        "Y_test_proba_pred": data.pop("Y_test_proba_pred")[rows_ok_test],
        **data,
    }


def load_data(candidate_path: Path):
    data = {}
    for s in tqdm(slices, desc="Loading data..."):
        data[s] = csr_array(
            pd.read_parquet((candidate_path / s).with_suffix(".parquet"))
        )
    return filter_no_labels(data)


def evaluate_data(data: Dict[str, pd.DataFrame | sparray | spmatrix | np.ndarray]):
    test_results = compute_metrics(
        y_true=data["Y_test"],
        y_pred=data["Y_test_pred"],
        y_score=data["Y_test_proba_pred"],
        k=5,
    )

    train_results = compute_metrics(
        y_true=data["Y_train"],
        y_pred=data["Y_train_pred"],
        y_score=data["Y_train_proba_pred"],
        k=5,
    )
    return train_results, test_results


if __name__ == "__main__":
    data = load_data(Path(sys.argv[1]))
    estimator = OmikujiClassifier()
    estimator.fit(
        X=data["X_train"].astype(np.float32), y=data["Y_train"].astype(np.uint32)
    )
    Y_test_pred_proba = estimator.predict_proba(data["X_test"])
    Y_test_pred = Y_test_pred_proba > 0.5

    with open("test_results_omikuji.json", "w") as output:
        json.dump(
            compute_metrics(
                y_true=data["Y_test"].astype(int),
                y_pred=Y_test_pred.astype(int).todense(),
                y_score=Y_test_pred_proba,
                k=5,
            ),
            output,
            indent=2,
        )
