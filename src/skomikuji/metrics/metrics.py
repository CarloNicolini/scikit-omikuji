from typing import Dict, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import sparray
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    precision_score,
    recall_score,
    zero_one_loss,
)

from skomikuji.metrics.xc import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

def validate_shapes(y_true, y_pred):
    if y_pred is not None:
        if y_true.shape != y_pred.shape:
            raise ValueError("Incompatbile shapes between arrays")
    
def validate_types(y_true, y_score, y_pred):
    if y_true.dtype != np.int64:
        raise ValueError("Convert y_true array to int")
    if y_pred.dtype != np.int64 and y_pred.dtype != bool:
        raise ValueError("Convert y_pred array to int")
    if y_score is not None:
        if y_score.dtype != np.float64:
            raise ValueError("Convert y_score array to float")


def compute_metrics(
    y_true: ArrayLike | sparray,
    y_pred: ArrayLike | sparray,
    y_score: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Computes the metrics for the model evaluation.

    Parameters
    ----------
    y_true: np.ndarray, scipy.sparse sparse array, pd.DataFrame
        The true labels.
    y_pred: np.ndarray, csr_matrix, pd.DataFrame
        The predicted labels.
    y_score: np.ndarary, sparse array, pd.DataFrame
        The predicted scores as from the method .predict_proba

    Returns
    -------
    Dict
        A dictionary containing the metrics.
    """

    validate_shapes(y_true, y_pred)
    validate_shapes(y_true, y_score)
    validate_shapes(y_pred, y_score)

    #validate_types(y_true, y_pred, y_score)

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
        if isinstance(y_score, sparray):
            y_score = y_score.todense()
        if "k" in kwargs:
            K = kwargs["k"]
            propensity_coeff = kwargs.get("propensity_coeff", None)
            for k in range(1, K + 1):
                all_metrics[f"ncdg@{k}"] = ndcg_at_k(y_true=y_true, y_pred=y_score, k=k)
                all_metrics[f"ncdg@{k}"] = ndcg_at_k(y_true=y_true, y_pred=y_score, k=k)
                all_metrics[f"precision@{k}"] = precision_at_k(
                    y_true=y_true, y_pred=y_score, k=k
                )
                all_metrics[f"recall@{k}"] = recall_at_k(
                    y_true=y_true, y_pred=y_score, k=k
                )
                all_metrics[f"f1@{k}"] = (
                    2
                    * all_metrics[f"recall@{k}"]
                    * all_metrics[f"precision@{k}"]
                    / (all_metrics[f"recall@{k}"] + all_metrics[f"precision@{k}"])
                )
                # when propensity coefficients also the propensity-scored metrics are computed
                if propensity_coeff is not None:
                    all_metrics[f"psncdg@{k}"] = ndcg_at_k(
                        y_true=y_true,
                        y_pred=y_score,
                        k=k,
                        propensity_coeff=propensity_coeff,
                    )
                    all_metrics[f"psprecision@{k}"] = precision_at_k(
                        y_true=y_true,
                        y_pred=y_score,
                        k=k,
                        propensity_coeff=propensity_coeff,
                    )
                    all_metrics[f"psrecall@{k}"] = recall_at_k(
                        y_true=y_true,
                        y_pred=y_score,
                        k=k,
                        propensity_coeff=propensity_coeff,
                    )
                    all_metrics[f"psf1@{k}"] = (
                        2
                        * all_metrics[f"psrecall@{k}"]
                        * all_metrics[f"psprecision@{k}"]
                        / (
                            all_metrics[f"psrecall@{k}"]
                            + all_metrics[f"psprecision@{k}"]
                        )
                    )

    if y_score is not None:
        all_metrics["log_loss"] = log_loss(
            y_true=y_true,
            y_pred=softmax(y_score, axis=1),
            sample_weight=sample_weight,
        )
    all_metrics["support"] = y_pred.shape[0]
    
    return all_metrics
