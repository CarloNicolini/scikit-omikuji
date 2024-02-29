from sklearn.metrics import make_scorer
from ._metrics import precision_at_k, recall_at_k, ndcg_at_k

precision_at_k_scorer = make_scorer(
    precision_at_k, greater_is_better=True, response_method="predict_proba"
)
recall_at_k_scorer = make_scorer(
    recall_at_k, greater_is_better=True, response_method="predict_proba"
)
ndcg_at_k_scorer = make_scorer(
    ndcg_at_k, greater_is_better=True, response_method="predict_proba"
)
