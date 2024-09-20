Metrics
=======

The metrics module provides various evaluation metrics and scoring functions for extreme multilabel classification tasks.

.. automodule:: skomikuji.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Metrics
------------------

.. autofunction:: precision_at_k

.. autofunction:: ndcg_at_k

.. autofunction:: recall_at_k

.. autofunction:: coverage_error

.. autofunction:: label_ranking_average_precision_score

.. autofunction:: label_ranking_loss

Scorer Functions
----------------

.. autofunction:: precision_at_k_scorer

.. autofunction:: ndcg_at_k_scorer

.. autofunction:: recall_at_k_scorer

.. autofunction:: coverage_error_scorer

.. autofunction:: label_ranking_average_precision_scorer

.. autofunction:: label_ranking_loss_scorer

Utility Functions
-----------------

.. autofunction:: make_scorer

.. autofunction:: _check_targets

.. autofunction:: _binary_clf_curve

XC Metrics
----------

.. automodule:: skomikuji.metrics.xc
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: precision_at_k_xc

.. autofunction:: ndcg_at_k_xc

.. autofunction:: recall_at_k_xc

.. autofunction:: propensity_score_xc

.. autofunction:: psprecision_at_k_xc

.. autofunction:: psndcg_at_k_xc

.. autofunction:: psrecall_at_k_xc
