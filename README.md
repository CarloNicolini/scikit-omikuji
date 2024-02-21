🏷️ scikit-omikuji
==============

This project contains a modified version of the Omikuji library for supporting `scipy.sparse.csr_matrix` directly fed in the input, without relying on expensive I/O operations and strange file formats.
Moreover we offer a scikit-learn compatible wrapper around the modified Omikuji 🏷️ library .

Omikuji is an efficient implementation of **PARABEL** (Partitioned Label Trees) by Prabhu et al., 2018 and its variations for extreme multi-label classification.
The underlying model code is written in Rust🦀 and its original version is offered by [Tom Tung](https://github.com/tomtung/omikuji).

## Features & Performance

Omikuji has has been tested on datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). All tests below are run on a quad-core Intel® Core™ i7-6700 CPU, and we allowed as many cores to be utilized as possible. We measured training time, and calculated precisions at 1, 3, and 5. (Note that, due to randomness, results might vary from run to run, especially for smaller datasets.)

### Parabel, better parallelized

Omikuji provides a more parallelized implementation of Parabel (Prabhu et al., 2018) that trains faster when more CPU cores are available. Compared to the [original implementation](http://manikvarma.org/code/Parabel/download.html) written in C++, which can only utilize the same number of CPU cores as the number of trees (3 by default), Omikuji maintains the same level of precision but trains 1.3x to 1.7x faster on our quad-core machine. **Further speed-up is possible if more CPU cores are available**.

| Dataset               | Metric        | Parabel       | Omikuji<br/>(balanced,<br/>cluster.k=2)       |
|-----------------      |------------   |---------      |------------------------------------------     |
|  EURLex-4K            | P@1           | 82.2          | 82.1                                          |
|                       | P@3           | 68.8          | 68.8                                          |
|                       | P@5           | 57.6          | 57.7                                          |
|                       | Train Time    | 18s           | 14s                                           |
| Amazon-670K           | P@1           | 44.9          | 44.8                                          |
|                       | P@3           | 39.8          | 39.8                                          |
|                       | P@5           | 36.0          | 36.0                                          |
|                       | Train Time    | 404s          | 234s                                          |
|  WikiLSHTC-325K       | P@1           | 65.0          | 64.8                                          |
|                       | P@3           | 43.2          | 43.1                                          |
|                       | P@5           | 32.0          | 32.1                                          |
|                       | Train Time    | 959s          | 659s                                          |

### Regular k-means for shallow trees

Following Bonsai (Khandagale et al., 2019), Omikuji supports using regular k-means instead of balanced 2-means clustering for tree construction, which results in wider, shallower and unbalanced trees that train slower but have better precision. Comparing to the [original Bonsai implementation](https://github.com/xmc-aalto/bonsai), Omikuji also achieves the same precisions while training 2.6x to 4.6x faster on our quad-core machine. (Similarly, further speed-up is possible if more CPU cores are available.)

| Dataset               | Metric        | Bonsai        | Omikuji<br/>(unbalanced,<br/>cluster.k=100,<br/>max\_depth=3) |
|-----------------      |------------   |---------      |-------------------------------------------------------------- |
|  EURLex-4K            | P@1           | 82.8          | 83.0                                                          |
|                       | P@3           | 69.4          | 69.5                                                          |
|                       | P@5           | 58.1          | 58.3                                                          |
|                       | Train Time    | 87s           | 19s                                                           |
| Amazon-670K           | P@1           | 45.5*         | 45.6                                                          |
|                       | P@3           | 40.3*         | 40.4                                                          |
|                       | P@5           | 36.5*         | 36.6                                                          |
|                       | Train Time    | 5,759s        | 1,753s                                                        |
|  WikiLSHTC-325K       | P@1           | 66.6*         | 66.6                                                          |
|                       | P@3           | 44.5*         | 44.4                                                          |
|                       | P@5           | 33.0*         | 33.0                                                          |
|                       | Train Time    | 11,156s       | 4,259s                                                        |

*\*Precision numbers as reported in the paper; our machine doesn't have enough memory to run the full prediction with their implementation.*

### Balanced k-means for balanced shallow trees

Sometimes it's desirable to have shallow and wide trees that are also balanced, in which case Omikuji supports the balanced k-means algorithm used by HOMER (Tsoumakas et al., 2008) for clustering as well.

| Dataset               | Metric        | Omikuji<br/>(balanced,<br/>cluster.k=100)     |
|-----------------      |------------   |------------------------------------------     |
|  EURLex-4K            | P@1           | 82.1                                          |
|                       | P@3           | 69.4                                          |
|                       | P@5           | 58.1                                          |
|                       | Train Time    | 19s                                           |
| Amazon-670K           | P@1           | 45.4                                          |
|                       | P@3           | 40.3                                          |
|                       | P@5           | 36.5                                          |
|                       | Train Time    | 1,153s                                        |
|  WikiLSHTC-325K       | P@1           | 65.6                                          |
|                       | P@3           | 43.6                                          |
|                       | P@5           | 32.5                                          |
|                       | Train Time    | 3,028s                                        |

### Layer collapsing for balanced shallow trees

An alternative way for building balanced, shallow and wide trees is to collapse adjacent layers, similar to the tree compression step used in AttentionXML (You et al., 2019): intermediate layers are removed, and their children replace them as the children of their parents. For example, with balanced 2-means clustering, if we collapse 5 layers after each layer, we can increase the tree arity from 2 to 2⁵⁺¹ = 64.

| Dataset               | Metric        | Omikuji<br/>(balanced,<br/>cluster.k=2,<br/>collapse 5 layers)        |
|-----------------      |------------   |---------------------------------------------------------------        |
|  EURLex-4K            | P@1           | 82.4                                                                  |
|                       | P@3           | 69.3                                                                  |
|                       | P@5           | 58.0                                                                  |
|                       | Train Time    | 16s                                                                   |
| Amazon-670K           | P@1           | 45.3                                                                  |
|                       | P@3           | 40.2                                                                  |
|                       | P@5           | 36.4                                                                  |
|                       | Train Time    | 460s                                                                  |
|  WikiLSHTC-325K       | P@1           | 64.9                                                                  |
|                       | P@3           | 43.3                                                                  |
|                       | P@5           | 32.3                                                                  |
|                       | Train Time    | 1,649s                                                                |


# Usage

The following script ([`tests/test-skomikuji.py`](tests/test-skomikuji.py)) demonstrates how to use the `scikit-omikuji` to train a model and make predictions on a fake dataset generated with `scikit-learn`.

```python
import numpy as np
import json
from skomikuji.models.skwrapper import OmikujiEstimator
from skomikuji.metrics import compute_metrics
from scipy.sparse import csr_matrix
from sklearn.datasets import make_multilabel_classification

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
model = OmikujiEstimator(top_k=10)
model = model.fit(X_train, Y_train)

### We get the class prediction probabilities and the predicted labels using 0.5 as the threshold
Y_proba_test_pred = model.predict_proba(X_test).todense()
Y_test_pred = Y_proba_test_pred > 0.5

# We compute some metrics, up to the precision@2 (precision@1 and precision@2 are reported)
print(
    json.dumps(
        compute_metrics(
            y_true=Y_test, y_pred=Y_test_pred, y_score=Y_proba_test_pred, k=2
        ),
        indent=4,
    )
)
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

## References
- Y. Prabhu, A. Kag, S. Harsola, R. Agrawal, and M. Varma, “Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising,” in Proceedings of the 2018 World Wide Web Conference, 2018, pp. 993–1002.
- S. Khandagale, H. Xiao, and R. Babbar, “Bonsai - Diverse and Shallow Trees for Extreme Multi-label Classification,” Apr. 2019.
- G. Tsoumakas, I. Katakis, and I. Vlahavas, “Effective and efficient multilabel classification in domains with large number of labels,” ECML, 2008.
- R. You, S. Dai, Z. Zhang, H. Mamitsuka, and S. Zhu, “AttentionXML: Extreme Multi-Label Text Classification with Multi-Label Attention Based Recurrent Neural Networks,” Jun. 2019.
