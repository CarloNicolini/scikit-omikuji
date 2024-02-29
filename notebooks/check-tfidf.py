import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.feature_extraction.text import TfidfVectorizer
from skomikuji import OmikujiClassifier
from skomikuji.metrics import compute_metrics

df = pd.read_parquet(
    "/data/datascience/smart_screening/experiments/v34/datasets/processed/candidates_full_after_bridge.parquet",
    columns=["CV_CONTENT_PROCESSED"],
)

Y = pd.read_parquet(
    "/data/datascience/smart_screening/experiments/v34/results/model_cv_interactions/Y.parquet",
)
Y_train = pd.read_parquet(
    "/data/datascience/smart_screening/experiments/v34/results/model_cv_interactions/Y_train.parquet",
)
Y_test = pd.read_parquet(
    "/data/datascience/smart_screening/experiments/v34/results/model_cv_interactions/Y_test.parquet",
)


cv = df["CV_CONTENT_PROCESSED"].dropna()
cv_train = pd.merge(left=cv, right=Y_train, on="CANDIDATE_ID", how="inner")
cv_test = pd.merge(left=cv, right=Y_test, on="CANDIDATE_ID", how="inner")

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    dtype=np.float32,
).fit(cv_train["CV_CONTENT_PROCESSED"].tolist())

X_train = tfidf_vectorizer.transform(cv_train["CV_CONTENT_PROCESSED"].tolist())
X_test = tfidf_vectorizer.transform(cv_test["CV_CONTENT_PROCESSED"].tolist())

Y_train = csr_array(Y.loc[cv_train.index, :], dtype=np.uint32)
Y_test = csr_array(Y.loc[cv_test.index, :], dtype=np.uint32)

model = OmikujiClassifier()
model.fit(X_train, Y_train)
model.fit(csr_array(X_train), csr_array(Y_train).astype(np.uint32))
Y_test_pred_proba = model.predict_proba(X_test)

pd.Series(
    compute_metrics(
        y_true=Y_test,
        y_pred=(Y_test_pred_proba > 0.5).astype(np.uint32),
        y_score=Y_test_pred_proba,
        k=5,
        propensity_coeff=(0.5, 0.4),
    )
)
