import numpy as np
from sklearn.datasets import make_multilabel_classification
from scipy.sparse import csr_array

def create_sparse_train_test_data():
    num_rows, num_features, num_labels = 1000, 100, 10
    np.random.seed(42)

    # Creates a fake dataset for multilabel classification. 1000 samples, 100 sparse features
    # Here we avoid any example with no labels
    X_train, Y_train = make_multilabel_classification(
        n_samples=num_rows,
        n_features=num_features,
        sparse=True,
        n_labels=num_labels,
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
        n_samples=num_rows//10,
        n_features=num_features,
        sparse=True,
        n_labels=num_labels,
        random_state=42,
        allow_unlabeled=False,
        n_classes=2,
        return_indicator="sparse",
    )
    X_test = Y_test.astype(np.float32)
    Y_test = Y_test.astype(np.uint32)
    
    return csr_array(X_train), csr_array(X_test), csr_array(Y_train), csr_array(Y_test)