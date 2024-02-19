import numpy as np
from typing import Dict, Optional, Tuple
from functools import partial
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
)
from sklearn.metrics import ndcg_score


# def inv_propensity(X_Y: np.ndarray, a_coeff: float = 0.5, b_coeff: float = 0.4):
#     """
#     function wts = inv_propensity(X_Y,A,B,varargin)

#         %% Returns inverse propensity weights
#         %% A,B are parameters of the propensity model. Refer to paper for more details.
#         %% A,B values used for different datasets in paper:
#         %%	Wikipedia-LSHTC: A=0.5,  B=0.4
#         %%	Amazon:          A=0.6,  B=2.6
#         %%	Other:			 A=0.55, B=1.5

#         num_inst = size(z,2);
#         freqs = sum(z,2);

#         C = (log(num_inst)-1)*(B+1)^A;
#         wts = 1 + C*(freqs+B).^-A;
#     end
#     """
#     n_cols = X_Y.shape[1]
#     freqs = np.sum(X_Y, axis=1)
#     C = (np.log(n_cols) - 1) * (b_coeff + 1) ** a_coeff
#     return np.power(1 + C * (freqs + b_coeff), -a_coeff)


# def precision_at_k(
#     y_true: np.ndarray,
#     y_score: np.ndarray,
#     k: int = 3,
#     propensity_array: Optional[np.ndarray] = None,
#     propensity_coeff: Optional[Tuple[float, float]] = None,
# ) -> float:
#     """
#     Returns the precision@k of an array.

#     Parameters
#     ----------
#     x: np.array
#         The array.
#     k: int
#         The number of indices to return.

#     Returns
#     -------
#     np.array
#         The top k indices.
#     """
#     y_true_ranked = np.take_along_axis(
#         arr=y_true, indices=np.argsort(y_score, axis=1)[:, ::-1], axis=1
#     )
#     if isinstance(propensity_coeff, (tuple,list)):
#         propensity_array = inv_propensity(y_true, *propensity_coeff)
#     if propensity_array is None:
#         return y_true_ranked[:, :k].mean()
#     elif isinstance(propensity_array, np.ndarray):
#         y_true_ranked = y_true_ranked / propensity_array
#     else:
#         raise TypeError("Propensity scores must either be None or a numpy array")


# def dcg_at_k(
#     y_true: np.ndarray,
#     y_score: np.ndarray,
#     k: int = 3,
#     log_base: int | float = 2,
#     normalized: bool = False,
# ) -> float:
#     """
#     Returns the discounted cumulative gain@k of an array.

#     Parameters
#     ----------
#     x: np.array
#         The array.
#     k: int
#         The number of indices to return.

#     Returns
#     -------
#     np.array
#         The top k indices.
#     """
#     indices = np.argsort(y_score, axis=1)[:, ::-1]
#     logs = np.log(indices + 2) / np.log(log_base)
#     y_true_ranked = np.take_along_axis(arr=y_true, indices=indices, axis=1)
#     dcg = (y_true_ranked / logs)[:, :k].sum() / k
#     if not normalized:
#         return dcg
#     else:
#         y_true_l0 = np.sum(y_true, axis=1)
#         y_true_l0 = np.tile(y_true_l0, y_true.shape[0]).reshape(y_true.shape)
#         denominator = np.take_along_axis(arr=np.power(logs, -1), indices=np.min(axis=1))
#         ndcg = dcg / denominator
#         return ndcg


# def top_k(x, k: int):
#     """
#     Returns the top k indices of an array.

#     Parameters
#     ----------
#     x: np.array
#         The array.
#     k: int
#         The number of indices to return.

#     Returns
#     -------
#     np.array
#         The top k indices.
#     """
#     ind = np.argpartition(x, -k)[-k:]
#     ind = ind[np.argsort(x[ind])]
#     ind.sort()
#     return ind
#
# def apk(y_true, y_pred, k=10):
#     """
#     Computes the average precision at k.

#     This function computes the average prescision at k between two lists of
#     items.

#     Parameters
#     ----------
#     actual : list
#              A list of elements that are to be predicted (order doesn't matter)
#     predicted : list
#                 A list of predicted elements (order does matter)
#     k : int, optional
#         The maximum number of predicted elements

#     Returns
#     -------
#     score : double
#             The average precision at k over the input lists

#     """
#     if len(y_pred) > k:
#         y_pred = y_pred[:k]

#     score = 0.0
#     num_hits = 0.0

#     for i, p in enumerate(y_pred):
#         if p in y_true and p not in y_pred[:i]:
#             num_hits += 1.0
#             score += num_hits / (i + 1.0)

#     if not y_true:
#         return 0.0

#     return score / min(len(y_true), k)


def compute_metrics(
    y_true, y_pred, y_score=None, sample_weight=None
) -> Dict[str, float]:
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
    all_metrics = {
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
    }

    if y_score is not None:
        all_metrics["ncdg_score"] = ndcg_score(y_true=y_true, y_score=y_score)
    all_metrics["log_loss"] = log_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    all_metrics["support"] = len(y_pred)
    
    return all_metrics

