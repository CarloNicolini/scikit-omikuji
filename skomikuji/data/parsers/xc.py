from pathlib import Path
from typing import List, Tuple
from scipy.sparse import csr_array

def parse_xc_repo_data_line(
    line: str, n_features: int
):
    tokens = line.split(" ")
    if ":" in tokens[0]:
        return False  # there is a misformatting in the file

    labels_str = tokens[0]
    labels = []
    for label_str in labels_str.split(","):
        if label_str.strip() and isinstance(label_str,str):
            try:
                label_int = int(label_str)
                labels.append(label_int)
            except ValueError:
                raise ValueError(f"Failed to parse label {label_str}")

    features = []
    for feature_value_pair_str in tokens[1:]:
        feature_value_pair = feature_value_pair_str.split(":")
        if len(feature_value_pair) != 2:
            raise ValueError(f"Failed to parse feature {feature_value_pair_str}")
        try:
            feature, value = feature_value_pair
        except ValueError:
            raise ValueError(f"Failed to parse feature value {feature_value_pair_str}")
        try:
            feature = int(feature)
            value = float(value)
        except Exception:
            raise ValueError(f"Cannot cast feature {feature} and value {value} to int and float")
        features.append((feature, value))

    features.sort()
    if not is_valid_sparse_vec(features, n_features):
        raise ValueError(f"Feature vector is invalid in line {line}")

    return features, labels



def is_valid_sparse_vec(features: List[Tuple[int, float]], n_features: int) -> bool:
    indices = {index for index, _ in features}
    return max(indices) < n_features


def read_xc_repo_file(path: Path):
    """

    Parameters
    ----------
    """
    if isinstance(path, Path):
        path = path.name

    with open(path, "r", encoding="utf-8") as f:
        num_samples, num_features, num_labels = [int(x) for x in f.readline().split(" ")]
        X, Y = [], []
        for line in f:
            x, y = parse_xc_repo_data_line(line=line, n_features=num_features)
            X.append(csr_array(x))
            Y.append(y)

    return X,Y
