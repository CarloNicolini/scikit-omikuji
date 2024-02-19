from typing import Optional, Tuple
import numpy as np
from scipy.sparse import csr_array, csr_matrix
from xclib.evaluation.xc_metrics import (
    precision,
    recall,
    ndcg,
    psprecision,
    psndcg,
    psrecall,
    compute_inv_propesity,
)
from sklearn.metrics import make_scorer


def precision_at_k(
    y_true: np.ndarray | csr_matrix | csr_array,
    y_pred: np.ndarray | csr_matrix | csr_array,
    k: int=1,
    propensity_array: Optional[np.ndarray] = None,
    propensity_coeff: Optional[Tuple[float, float]] = None,
    sorted: bool = False,
) -> float:
    """
    Returns the precision@k.

    Parameters
    ----------
    Y_true: np.ndarray, csr_matrix, dict
        The 2D array of ground truth labels.
    Y_score: csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}

    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients

    k: int
        The number of indices to return.
    """
    if isinstance(propensity_coeff, (tuple, list)):
        propensity_array = compute_inv_propesity(
            labels=y_true, A=propensity_coeff[0], B=propensity_coeff[1]
        )
    if isinstance(propensity_array, np.ndarray):
        return psprecision(
            X=y_pred, true_labels=y_true, inv_psp=propensity_array, k=k, sorted=sorted
        )[-1]
    elif propensity_array is None:
        return precision(X=y_pred, true_labels=y_true, k=k, sorted=sorted)[-1]
    else:
        raise ValueError("Unsupported propensity array type")


def recall_at_k(
    y_true: np.ndarray | csr_matrix | csr_array,
    y_pred: np.ndarray | csr_matrix | csr_array,
    k: int=1,
    propensity_array: Optional[np.ndarray] = None,
    propensity_coeff: Optional[Tuple[float, float]] = None,
    sorted: bool = False,
) -> float:
    """
    Returns the recall@k.

    Parameters
    ----------
    Y_true: np.ndarray, csr_matrix, dict
        The 2D array of ground truth labels.
    Y_score: csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}

    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients

    k: int
        The number of indices to return.
    """
    if isinstance(propensity_coeff, (tuple, list)):
        propensity_array = compute_inv_propesity(
            labels=y_true, A=propensity_coeff[0], B=propensity_coeff[1]
        )
    if isinstance(propensity_array, np.ndarray):
        return psrecall(
            X=y_pred, true_labels=y_true, inv_psp=propensity_array, k=k, sorted=sorted
        )[-1]
    elif propensity_array is None:
        return recall(X=y_pred, true_labels=y_true, k=k, sorted=sorted)[-1]
    else:
        raise ValueError("Unsupported propensity array type")


def ndcg_at_k(
    y_true: np.ndarray | csr_matrix | csr_array,
    y_pred: np.ndarray | csr_matrix | csr_array,
    k: int=1,
    propensity_array: Optional[np.ndarray] = None,
    propensity_coeff: Optional[Tuple[float, float]] = None,
    sorted: bool = False,
) -> float:
    """
    Returns the normalized cumulative discount gain@k.

    Parameters
    ----------
    Y_true: np.ndarray, csr_matrix, dict
        The 2D array of ground truth labels.
    Y_score: csr_matrix, np.ndarray or dict
        The 2D array of labels relevance as found by the classifier .predict_proba
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}

    propensity_array:
        An array with the inverse propensity scores
    propensity_coeff:
        A tuple with two elements representing the propensity coefficients

    k: int
        The number of indices to return.
    """
    if isinstance(propensity_coeff, (tuple, list)):
        propensity_array = compute_inv_propesity(
            labels=y_true, A=propensity_coeff[0], B=propensity_coeff[1]
        )
    if isinstance(propensity_array, np.ndarray):
        return psndcg(
            X=y_pred, true_labels=y_true, inv_psp=propensity_array, k=k, sorted=sorted
        )[-1]
    elif propensity_array is None:
        return ndcg(X=y_pred, true_labels=y_true, k=k, sorted=sorted)[-1]
    else:
        raise ValueError("Unsupported propensity array type")


precision_at_k_scorer = make_scorer(
    precision_at_k, greater_is_better=True
)
recall_at_k_scorer = make_scorer(
    recall_at_k, greater_is_better=True
)
ndcg_at_k_scorer = make_scorer(
    precision_at_k, greater_is_better=True
)
