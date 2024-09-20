Ensemble
========

The ensemble module provides functionality for creating and managing ensembles of Omikuji models for extreme multilabel classification.

.. automodule:: skomikuji._omikuji
   :members:
   :undoc-members:
   :show-inheritance:

OmikujiEnsemble
---------------

.. autoclass:: OmikujiEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: fit

   .. automethod:: predict

   .. automethod:: predict_proba

Helper Functions
----------------

.. autofunction:: _create_omikuji_model

.. autofunction:: _fit_omikuji_model

.. autofunction:: _predict_omikuji_model

.. autofunction:: _predict_proba_omikuji_model

Utility Functions
-----------------

.. autofunction:: get_omikuji_version

.. autofunction:: set_num_threads
