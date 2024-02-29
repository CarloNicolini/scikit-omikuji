import numpy as np
import pytest
from scipy.sparse import sparray, csr_array
from skomikuji.data.dataset import create_sparse_train_test_data
from skomikuji.ensemble import OmikujiClassifier


@pytest.fixture
def test_dataset():
    # Creates a fake dataset for multilabel classification. 1000 samples, 100 sparse features
    # Here we avoid any example with no labels
    yield create_sparse_train_test_data


def test_instantiation_with_default_parameters():
    estimator = OmikujiClassifier()
    assert isinstance(estimator, OmikujiClassifier)
    assert estimator.top_k == 5
    assert estimator.beam_size == 10
    assert estimator.n_trees == 3
    assert estimator.min_branch_size == 100
    assert estimator.max_depth == 20
    assert estimator.centroid_threshold == 0.0
    assert estimator.collapse_every_n_layers == 0
    assert estimator.linear_eps == 0.1
    assert estimator.linear_c == 1
    assert estimator.linear_weight_threshold == 0.1
    assert estimator.linear_max_iter == 20
    assert estimator.cluster_k == 2
    assert estimator.cluster_balanced is False
    assert estimator.cluster_eps == 0.0001
    assert estimator.cluster_min_size == 2
    assert estimator.n_jobs is None


def test_instantiation_with_custom_parameters():
    estimator = OmikujiClassifier(
        top_k=5,
        beam_size=20,
        n_trees=5,
        min_branch_size=200,
        max_depth=30,
        centroid_threshold=0.5,
        collapse_every_n_layers=2,
        linear_eps=0.2,
        linear_c=2,
        linear_weight_threshold=0.2,
        linear_max_iter=30,
        cluster_k=3,
        cluster_balanced=True,
        cluster_eps=0.0002,
        cluster_min_size=3,
        n_jobs=4,
    )
    assert isinstance(estimator, OmikujiClassifier)
    assert estimator.top_k == 5
    assert estimator.beam_size == 20
    assert estimator.n_trees == 5
    assert estimator.min_branch_size == 200
    assert estimator.max_depth == 30
    assert estimator.centroid_threshold == 0.5
    assert estimator.collapse_every_n_layers == 2
    assert estimator.linear_eps == 0.2
    assert estimator.linear_c == 2
    assert estimator.linear_weight_threshold == 0.2
    assert estimator.linear_max_iter == 30
    assert estimator.cluster_k == 3
    assert estimator.cluster_balanced is True
    assert estimator.cluster_eps == 0.0002
    assert estimator.cluster_min_size == 3
    assert estimator.n_jobs == 4



def test_X_dtype_error():
    estimator = OmikujiClassifier()
    X = csr_array(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),dtype=np.float32)
    Y = csr_array(np.array([[1, 0, 1], [0, 1, 0]]),dtype=np.uint32)
    estimator.fit(
        X,Y
    )
    Y_proba_pred = estimator.predict_proba(X)


def test_fit_estimator(test_dataset):
    X_train, X_test, Y_train, Y_test = test_dataset()
    estimator = OmikujiClassifier()
    estimator = estimator.fit(X_train, Y_train)
    Y_test_proba_pred = estimator.predict_proba(X_test)
    assert Y_test_proba_pred.shape == Y_test.shape
