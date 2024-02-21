from skomikuji.models.skwrapper import OmikujiEstimator
from skomikuji.data.parsers.svmlight import load_train_test_data_svm
from skomikuji.metrics import compute_metrics
from sklearn.metrics import multilabel_confusion_matrix

model = OmikujiEstimator(top_k=5)

data = load_train_test_data_svm(
    "/home/ubuntu/workspace/skomikuji/data/eurlex/eurlex_train.txt",
    "/home/ubuntu/workspace/skomikuji/data/eurlex/eurlex_test.txt",
)
print(data)

""" data = {}
data["X_train"] = csr_array(np.random.randn(10, 3) * (np.random.randn(10, 3) > -1.0))
data["Y_train"] = csr_array(np.random.randn(10, 3) > -1.0)
print(data["Y_train"].shape)
data["X_test"] = csr_array(np.random.randn(5, 3) * (np.random.randn(5, 3) > -1.0))
data["Y_test"] = csr_array(np.random.randn(5, 3) > -1.0) """
model = model.fit(data["X_train"], data["Y_train"])

Y_proba_test_pred = model.predict_proba(data["X_test"]).todense()
Y_test_pred = Y_proba_test_pred > 0.5

print(compute_metrics(data["Y_test"], Y_test_pred))
print(multilabel_confusion_matrix(data["Y_test"], Y_test_pred))
