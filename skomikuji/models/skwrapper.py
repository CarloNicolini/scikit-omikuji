import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

import numpy as np
import omikuji as omi
from numba import njit
from numpy.typing import ArrayLike
from scipy.sparse import csr_array, vstack
from sklearn.base import BaseEstimator
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file as load_svmlight_file_sklearn
from sklearn.metrics import multilabel_confusion_matrix
from tabulate import tabulate

from skomikuji.data.parsers.svmlight import load_svm_light
from skomikuji.metrics import compute_metrics

from skomikuji.data.parsers.xc import read_xc_repo_file

logger = logging.getLogger()


def line_prepender(filename, line):
    with open(filename, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip("\r\n") + "\n" + content)


def load_svm_light_file_patch(path):
    i = 0
    x_row, x_cols, x_values = [], [], []
    Y = []
    for line in Path(path).read_text().split("\n"):
        if i == 0:
            n_samples, n_features, n_labels = [int(i) for i in line.split()]
            i += 1
            continue
        if line:
            y, *x = line.split(" ")
            if ":" in y:  # it's because the label was missing!
                continue
            try:
                y = [int(i) for i in y.split(",")]
            except Exception:
                # print(f"Error at line {i}")
                continue
            Y.append(y)
            for r in x:
                feat, val = r.split(":")
                feat = int(feat)
                val = float(val)
                x_row.append(i - 1)
                x_cols.append(feat)
                x_values.append(val)
            i += 1

    X = csr_array((x_values, (x_row, x_cols)), shape=(i-1, n_features))
    rowcols = [(row, col) for row, labels in enumerate(Y) for col in labels]
    rows = [r[0] for r in rowcols]
    cols = [r[1] for r in rowcols]
    Y = csr_array((len(rows) * [1], (rows, cols)), shape=(i, n_labels))
    return X, Y


class OmikujiEstimator(BaseEstimator):
    """
    Efficient Rust implementation of Parabel algorithm
    http://manikvarma.org/pubs/prabhu18b.pdf


    Parameters
    ----------
    Parabel has 4 hyperparameters:
        (a) the number of trees (T );
        (b) the maximum number of paths that can be traversed in a tree (P); (c) the maximum number of labels in a leaf node (M) and
        (d) the misclassification penalty for all the internal and
        leaf node classifiers (C).
        The default parameter settings of M = 100, P = 10 and C = 10 (C = 1) for log loss (squared hinge loss) were used in all the experiments to eliminate hyperparameter sweeps (though tuning could have increased Parabel's accuracy).

    """

    def __init__(
        self,
        top_k: int = 3,
        beam_size: int = 10,
        n_trees: int = 3,
        min_branch_size: int = 100,
        max_depth: int = 20,
        centroid_threshold: float = 0.0,
        collapse_every_n_layers: int = 0,
        linear_eps: float = 0.1,
        linear_c: float = 1,
        linear_weight_threshold: float = 0.1,
        linear_max_iter: int = 20,
        cluster_k: int = 2,
        cluster_balanced: bool = False,
        cluster_eps: float = 0.0001,
        cluster_min_size: int = 2,
    ) -> None:
        self.beam_size = beam_size
        self.top_k = top_k
        self.min_branch_size = min_branch_size
        self.max_depth = max_depth
        self.centroid_threshold = centroid_threshold
        self.collapse_every_n_layers = collapse_every_n_layers
        self.linear_eps = linear_eps
        self.linear_c = linear_c
        self.linear_weight_threshold = linear_weight_threshold
        self.linear_max_iter = linear_max_iter
        self.cluster_k = cluster_k
        self.cluster_balanced = cluster_balanced
        self.cluster_eps = cluster_eps
        self.cluster_min_size = cluster_min_size

    @staticmethod
    def _train_omikuji_from_X_Y_dense_arrays(X, Y, hyper_param):
        with NamedTemporaryFile(delete=False, mode="w", prefix="omikuji-") as tmp_file:
            dump_svmlight_file(
                X=X, y=Y, f=tmp_file.name, multilabel=True, zero_based=True
            )
            line_prepender(
                tmp_file.name,
                f"{X.shape[0]} {X.shape[1]} {Y.shape[1]}\n",
            )
            return omi.Model.train_on_data(tmp_file.name, hyper_param)

    def validate_features(self, X: ArrayLike | csr_array):
        assert isinstance(X, (np.ndarray,csr_array)),"Only ndarray or csr_matrix accepted"
        S = X.sum(axis=1) != 0
        assert S.all(), "Some examples have no nonzero features"

    def validate_labels(self, Y: ArrayLike | csr_array):
        assert isinstance(
            Y, (np.ndarray, csr_array)
        ), "Only ndarray or csr_matrix accepted"
        S = Y.sum(axis=1) != 0
        assert S.all(), "Some examples have no label"

    def fit(
        self, X: ArrayLike | csr_array, Y: ArrayLike | csr_array, **fit_params
    ) -> "OmikujiEstimator":
        """
        Parameters
        ----------
        X: ArrayLike | csr_array
            Either a dense or a sparse compressed row array with features.
            It has num_samples x num_features shape.
        Y: ArrayLike | csr_array
            Either a dense 0/1 or a sparse compressed row array with labels
            It has num_samples x num_labels shape.

        Returns
        -------
        OmikujiEstimator instance
        """
        self.validate_features(X)
        self.validate_labels(Y)
        hyper_param = omi.Model.default_hyper_param()
        # applies the hyperparameters
        hyper_param.min_branch_size = self.min_branch_size
        hyper_param.max_depth = self.max_depth
        hyper_param.centroid_threshold = self.centroid_threshold
        hyper_param.collapse_every_n_layers = self.collapse_every_n_layers
        hyper_param.linear_eps = self.linear_eps
        hyper_param.linear_c = self.linear_c
        hyper_param.linear_weight_threshold = self.linear_weight_threshold
        hyper_param.linear_max_iter = self.linear_max_iter
        hyper_param.cluster_k = self.cluster_k
        hyper_param.cluster_balanced = self.cluster_balanced
        hyper_param.cluster_eps = self.cluster_eps
        hyper_param.cluster_min_size = self.cluster_min_size
        
        self.num_labels_ = Y.shape[1]
        if isinstance(X, (np.ndarray, csr_array)) and isinstance(
            Y, (np.ndarray, csr_array)
        ):
            self.model_ = OmikujiEstimator._train_omikuji_from_X_Y_dense_arrays(
                X, Y, hyper_param=hyper_param
            )
        return self

    def predict_proba(self, X: ArrayLike | csr_array) -> csr_array:
        """
        Predict new values one by one.
        Unfortunately this is the way to call omikuji, sample by sample
        """
        self.validate_features(X)
        num_samples = X.shape[0]
        if isinstance(X, csr_array):
            # we still deal with generators instead of instantiating things
            feature_value_pairs = (zip(x.indices, x.data) for x in X)
        elif isinstance(X, np.ndarray):
            feature_value_pairs = (
                ((i, v) for i, v in zip(np.where(X[0, :])[0], x[x != 0])) for x in X
            )

        row_indices, col_indices, values = [], [], []
        for index, sample_feat_value_pairs in enumerate(feature_value_pairs):
            # TODO check the proba of scores
            y_pred = self.model_.predict(
                sample_feat_value_pairs, beam_size=self.beam_size, top_k=self.top_k
            )
            for feat_index, feat_proba in y_pred:
                row_indices.append(index)
                col_indices.append(feat_index)
                values.append(feat_proba)

        Y_proba = csr_array(
            (values, (row_indices, col_indices)), shape=(num_samples, self.num_labels_)
        )
        return Y_proba

    def predict(
        self, X: ArrayLike | csr_array, proba_threshold: float = 0.5
    ) -> csr_array:
        return (self.predict_proba(X) > 0.5).astype(int)


def load_train_test_data_svm(
    train_data_path: str, test_data_path: str
) -> Dict[str, csr_array]:
    num_samples_train, num_features_train, num_labels_train = [
        int(x) for x in open(train_data_path, "r").readline().split()
    ]
    num_samples_test, num_features_test, num_labels_test = [
        int(x) for x in open(train_data_path, "r").readline().split()
    ]
    assert (
        num_features_test == num_features_train
    ), "Incompatible number of features between train and test"
    assert (
        num_labels_test == num_labels_train
    ), "Incompatible number of labels between train and test"
    
    X_train, Y_train = load_svm_light_file_patch(train_data_path)
    X_test, Y_test = load_svm_light_file_patch(test_data_path)
    return {"X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test}


if __name__ == "__main__":
    model = OmikujiEstimator(top_k=5)
    
    data = load_train_test_data_svm(
        "/home/ubuntu/workspace/skomikuji/data/eurlex/eurlex_train.txt",
        "/home/ubuntu/workspace/skomikuji/data/eurlex/eurlex_test.txt",
    )
    print(data)
    
    """ data = {}
    data["X_train"] = csr_array(np.random.randn(10, 3) * (np.random.randn(10, 3) > -1.0))
    data["Y_train"] = csr_array(np.random.randn(10, 3) > -1.0)
    print(data["Y_train"].shape)
    data["X_test"] = csr_array(np.random.randn(5, 3) * (np.random.randn(5, 3) > -1.0))
    data["Y_test"] = csr_array(np.random.randn(5, 3) > -1.0) """
    model = model.fit(data["X_train"], data["Y_train"])

    Y_proba_test_pred = model.predict_proba(data["X_test"]).todense()
    Y_test_pred = Y_proba_test_pred > 0.5

    print(compute_metrics(data["Y_test"], Y_test_pred))
    print(multilabel_confusion_matrix(data["Y_test"], Y_test_pred))
