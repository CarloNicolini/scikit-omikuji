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

# We compute some metrics
print(
    json.dumps(
        compute_metrics(
            y_true=Y_test, y_pred=Y_test_pred, y_score=Y_proba_test_pred,k=2
        ),
        indent=4,
    )
)
