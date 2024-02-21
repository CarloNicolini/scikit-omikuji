import contextlib
import logging
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np
import omikuji as omi
from numpy.typing import ArrayLike
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator
from sklearn.datasets import dump_svmlight_file

from skomikuji.data.parsers.svmlight import line_prepender

logger = logging.getLogger()


class OmikujiEstimator(BaseEstimator):
    """
    Efficient Rust implementation of Parabel algorithm
    http://manikvarma.org/pubs/prabhu18b.pdf


    Parameters
    ----------
    Parabel has 4 hyperparameters:
        (a) the number of trees (T );
        (b) the maximum number of paths that can be traversed in a tree (P); 
        (c) the maximum number of labels in a leaf node (M);
        (d) the misclassification penalty for all the internal and leaf node classifiers (C)
        .
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
        n_jobs: Optional[int] = None
    ) -> None:
        self.top_k = top_k
        self.beam_size = beam_size
        self.n_trees = n_trees
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
        self.n_jobs = n_jobs

    @staticmethod
    def _train_omikuji_from_X_Y_dense_arrays(X, Y, hyper_param, n_jobs=None):
        """
        Temporarily dump the arrays to disk in XC repo format and load them 
        using the Rust function train_on_data
        Parameters
        ----------
        """
        with NamedTemporaryFile(delete=True, mode="w", prefix="omikuji-") as tmp_file:
            dump_svmlight_file(
                X=X, y=Y, f=tmp_file.name, multilabel=True, zero_based=True
            )
            line_prepender(
                tmp_file.name,
                f"{X.shape[0]} {X.shape[1]} {Y.shape[1]}\n",
            )
            return omi.Model.train_on_data(tmp_file.name, hyper_param, n_threads=n_jobs)
        
    @staticmethod
    def _train_omikuji_from_X_Y_sparse_arrays(X: csr_array, Y: csr_array, hyper_param=None, n_jobs=None):
        """
        Pass the sparse arrays directly to underlying Rust implementation
        Parameters
        ----------
        """
        return omi.Model.train_on_features_labels(features=X, labels=Y, hyper_param=hyper_param, n_threads=n_jobs)

    def validate_features(self, X: ArrayLike | csr_array):
        assert isinstance(X, (np.ndarray,csr_array)),"Only ndarray or csr_array accepted"
        S = X.sum(axis=1) != 0
        assert S.all(), "Some examples have no nonzero features"

    def validate_labels(self, Y: ArrayLike | csr_array):
        assert isinstance(
            Y, (np.ndarray, csr_array)
        ), "Only ndarray or csr_array accepted"
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
        hyper_param.n_trees = self.n_trees
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
            self.model_ = OmikujiEstimator._train_omikuji_from_X_Y_sparse_arrays(
                X, Y, hyper_param=hyper_param, n_jobs=self.n_jobs
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
