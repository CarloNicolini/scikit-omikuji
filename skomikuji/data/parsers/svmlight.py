from pathlib import Path
from typing import Dict
from scipy.sparse import csr_array


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


def load_svm_light_file_patch(path: str | Path):
    """
    Parameters
    ----------
    path: str, Path
    """
    i = 0
    x_row, x_cols, x_values = [], [], []
    Y = []
    for line in Path(path).read_text().split("\n"):
        if i == 0:
            n_samples, n_features, n_labels = [int(i) for i in line.split()]
            i += 1
            continue
        if line:
            y, *x = line.split(" ")
            if ":" in y:  # it's because the label was missing!
                continue
            try:
                y = [int(i) for i in y.split(",")]
            except Exception:
                # print(f"Error at line {i}")
                continue
            Y.append(y)
            for r in x:
                feat, val = r.split(":")
                feat = int(feat)
                val = float(val)
                x_row.append(i - 1)
                x_cols.append(feat)
                x_values.append(val)
            i += 1

    X = csr_array((x_values, (x_row, x_cols)), shape=(i-1, n_features))
    rowcols = [(row, col) for row, labels in enumerate(Y) for col in labels]
    rows = [r[0] for r in rowcols]
    cols = [r[1] for r in rowcols]
    Y = csr_array((len(rows) * [1], (rows, cols)), shape=(i, n_labels))
    return X, Y


def line_prepender(filename: str, line: str):
    """
    Add a line on top of a file and saves it back
    """
    with open(filename, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip("\r\n") + "\n" + content)


def load_train_test_data_svm(
    train_data_path: str, test_data_path: str
) -> Dict[str, csr_array]:
    """
    Load the training and test data from two files in XC repo format
    """
    num_samples_train, num_features_train, num_labels_train = [
        int(x) for x in open(train_data_path, "r").readline().split()
    ]
    num_samples_test, num_features_test, num_labels_test = [
        int(x) for x in open(train_data_path, "r").readline().split()
    ]
    assert (
        num_features_test == num_features_train
    ), "Incompatible number of features between train and test"
    assert (
        num_labels_test == num_labels_train
    ), "Incompatible number of labels between train and test"

    X_train, Y_train = load_svm_light_file_patch(train_data_path)
    X_test, Y_test = load_svm_light_file_patch(test_data_path)
    return {"X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test}
