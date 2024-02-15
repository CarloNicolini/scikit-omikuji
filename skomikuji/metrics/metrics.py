import numpy as np
from scipy.sparse import csr_matrix
from numba import jit, njit
from typing import Dict, List, Any 

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    zero_one_loss,
    log_loss,
    make_scorer,
    multilabel_confusion_matrix,
)

def precision_at_k(Y_true: np.ndarray, Y_pred: np.ndarray, K: int) -> np.ndarray:
    """
    How many relevant items are present in the top-k recommendations of your system
    """
    @jit(nopython=True)
    def helper(score_mat: np.ndarray, true_mat: np.ndarray) -> np.ndarray:
        num_inst = score_mat.shape[1]
        num_lbl = score_mat.shape[0]

        P = np.zeros(K)
        rank_mat = np.argsort(-score_mat, axis=0) + 1  # Sort scores in descending order

        for k in range(1, K + 1):
            mat = rank_mat.copy()
            mat[mat > k] = 0
            mat = np.where(mat > 0, 1, 0)
            mat = mat * true_mat
            num = np.sum(mat, axis=1)

            P[k - 1] = np.mean(num / k)

        return P

    return helper(Y_pred, Y_true)


@njit
def ndcg_k(Y_true: np.ndarray, Y_pred: np.ndarray, K: int) -> np.ndarray:
    num_instances = Y_true.shape[0]
    num_labels = Y_true.shape[1]

    # Calculate weights
    weights = 1 / np.log2(np.arange(num_labels) + 2)
    cumulative_weights = np.cumsum(weights)

    # Sort predicted scores and true labels
    sorted_indices = np.argsort(-Y_pred, axis=1)
    sorted_scores = -np.sort(-Y_pred, axis=1)
    sorted_true_labels = np.take_along_axis(Y_true, sorted_indices, axis=1)

    ndcg_values = np.zeros(K)

    for k in range(1, K + 1):
        discounted_gain = 1 / np.log2(sorted_indices[:, :k] + 2)
        cumulative_discounted_gain = np.cumsum(discounted_gain, axis=1)
        cumulative_discounted_gain = np.pad(
            cumulative_discounted_gain, ((0, 0), (0, K - k)), mode="edge"
        )

        relevant_count = np.minimum(np.sum(sorted_true_labels[:, :k], axis=1), k)
        denominator = cumulative_weights[relevant_count - 1]

        ndcg_values[k - 1] = np.mean(
            np.sum(cumulative_discounted_gain / denominator[:, np.newaxis], axis=0)
        )

    return ndcg_values


def compute_metrics(y_true, y_pred, sample_weight=None) -> Dict[str, float]:
    """
    Computes the metrics for the model evaluation.

    Parameters
    ----------
    y_true: pd.DataFrame
        The true labels.
    y_pred: pd.DataFrame
        The predicted labels.

    Returns
    -------
    Dict
        A dictionary containing the metrics.
    """
    return {
        "precision_weighted": precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "precision_micro": precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "recall_weighted": recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "recall_micro": recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "f1_weighted": f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "f1_micro": f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "accuracy": accuracy_score(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        ),
        "hamming_loss": hamming_loss(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        ),
        "zero_one_loss": zero_one_loss(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        ),
        "jaccard_score_weighted": jaccard_score(
            y_true=y_true,
            y_pred=y_pred,
            average="weighted",
            zero_division=0.0,
            sample_weight=sample_weight,
        ),
        "jaccard_score_micro": jaccard_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            zero_division=0.0,
            sample_weight=sample_weight,
        ),
        "log_loss": log_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
        "support": len(y_pred),
    }


def apk(y_true, y_pred, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not y_true:
        return 0.0

    return score / min(len(y_true), k)


def mapk(y_true, y_pred, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])
