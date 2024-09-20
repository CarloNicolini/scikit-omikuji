🏷️ scikit-omikuji
==============

This project contains a modified version of the Omikuji library for supporting `scipy.sparse.csr_matrix` directly fed in the input, without relying on expensive I/O operations and strange file formats.
Moreover we offer a scikit-learn compatible wrapper around the modified Omikuji 🏷️ library .

Omikuji is an efficient implementation of **PARABEL** (Partitioned Label Trees) by Prabhu et al., 2018 and its variations for extreme multi-label classification.
The underlying model code is written in Rust🦀 and its original version is offered by [Tom Tung](https://github.com/tomtung/omikuji).


# Installation
Create a conda environment based on Python >= 3.10.

>>> pip install -e .


# Usage

The following script ([`tests/test-skomikuji.py`](tests/test-skomikuji.py)) demonstrates how to use the `scikit-omikuji` to train a model and make predictions on a fake dataset generated with `scikit-learn`.

```python
import numpy as np
import json
from skomikuji import OmikujiClassifier
from skomikuji.metrics import compute_metrics
from scipy.sparse import csr_array
from sklearn.datasets import make_multilabel_classification
import rich

num_rows, num_features, num_labels = 100, 10, 5
np.random.seed(42)

# Creates a fake dataset for multilabel classification. 1000 samples, 100 sparse features
# Here we avoid any example with no labels
X_train, Y_train = make_multilabel_classification(
    n_samples=1000,
    n_features=100,
    sparse=True,
    n_labels=10,
    random_state=42,
    allow_unlabeled=False,
    n_classes=2,
    return_indicator="sparse",
)
# Omikuji only allow for features in sparse format in np.float32 type
X_train = X_train.astype(np.float32)
# Omikuji only allow for labels in sparse format in np.uint32 type
Y_train = Y_train.astype(np.uint32)

# we also define a test dataset with the same parameters
X_test, Y_test = make_multilabel_classification(
    n_samples=100,
    n_features=100,
    sparse=True,
    n_labels=10,
    random_state=42,
    allow_unlabeled=False,
    n_classes=2,
    return_indicator="sparse",
)
X_test = Y_test.astype(np.float32)
Y_test = Y_test.astype(np.uint32)

### We now fit the Omikuji estimator on data
model = OmikujiClassifier(top_k=10)
model = model.fit(X_train, Y_train)

### We get the class prediction probabilities and the predicted labels using 0.5 as the threshold
Y_proba_test_pred = model.predict_proba(X_test)
Y_test_pred = Y_proba_test_pred > 0.5

# We compute some metrics, up to the precision@2 (precision@1 and precision@2 are reported)
rich.print_json(data=compute_metrics(y_true=Y_test, y_pred=Y_test_pred, y_score=Y_proba_test_pred, k=2))
```

On a Linux system, the result is the following:


| Metric                 | Value      |
|:-----------------------|-----------:|
| precision_weighted     |   0.911758 |
| precision_micro        |   0.91     |
| recall_weighted        |   1        |
| recall_micro           |   1        |
| f1_weighted            |   0.953384 |
| f1_micro               |   0.95288  |
| accuracy               |   0.82     |
| hamming_loss           |   0.09     |
| zero_one_loss          |   0.18     |
| jaccard_score_weighted |   0.911758 |
| jaccard_score_micro    |   0.91     |
| ncdg@1                 |   0.95     |
| precision@1            |   0.95     |
| recall@1               |   0.54     |
| f1@1                   |   0.688591 |
| ncdg@2                 |   1        |
| precision@2            |   0.91     |
| recall@2               |   1        |
| f1@2                   |   0.95288  |
| log_loss               |   1.26132  |
| support                | 100        |


### Datatypes
It's important to remember to maintain the `np.float32` datatype for the features and `np.uint32` for the labels.
Mispecification of these types results in a `TypeError`. This is because we employ very efficient memory mapping for sharing the memory between the Python numpy arrays and the underlying Rust context.


# Hyperparameters tuning
The main hyperparameters of the `OmikujiEstimator` are three.
- `n_trees`
- `max_depth`
- `linear_c`


## References
- Y. Prabhu, A. Kag, S. Harsola, R. Agrawal, and M. Varma, “Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising,” in Proceedings of the 2018 World Wide Web Conference, 2018, pp. 993–1002.
- S. Khandagale, H. Xiao, and R. Babbar, “Bonsai - Diverse and Shallow Trees for Extreme Multi-label Classification,” Apr. 2019.
- G. Tsoumakas, I. Katakis, and I. Vlahavas, “Effective and efficient multilabel classification in domains with large number of labels,” ECML, 2008.
- R. You, S. Dai, Z. Zhang, H. Mamitsuka, and S. Zhu, “AttentionXML: Extreme Multi-Label Text Classification with Multi-Label Attention Based Recurrent Neural Networks,” Jun. 2019.

## Documentation

To build and serve the documentation locally:

1. Install the required dependencies:
