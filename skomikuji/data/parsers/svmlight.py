from pathlib import Path
from scipy.sparse import csr_array
from sklearn.datasets import load_svmlight_file as load_svmlight_file_sklearn

def load_svm_light(path):
    i = 0
    x_row, x_cols, x_values = [], [], []
    Y = []
    for line in Path(path).read_text().split("\n"):
        if i == 0:
            n_samples, n_features, n_labels = [int(i) for i in line.split()]
        if line:
            y, *x = line.split()
            y = [int(i) for i in y.split(",")]
            Y.append(y)
            print(line)
            for r in x:
                feat, val = r.split(":")
                feat = int(feat)
                val = float(val)
                x_row.append(i)
                x_cols.append(feat)
                x_values.append(val)
            i += 1
    X = csr_array((x_row, x_cols, x_values), shape=(n_samples, n_features))
    Y_rows = [row for row, labels in enumerate(Y) for lab in row]
    Y_cols = [lab for row, labels in enumerate(Y) for lab in row]
    Y_vals = [1 for row, labels in enumerate(Y) for lab in row]
    Y = csr_array((Y_rows, Y_cols, Y_vals), shape=(n_samples, n_labels))
    return X, Y
