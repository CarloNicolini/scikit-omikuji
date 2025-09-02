# 🏷️ Scikit-Omikuji

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/scikit-omikuji/badge/?version=latest)](https://scikit-omikuji.readthedocs.io/en/latest/?badge=latest)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Scikit-learn compatible wrapper for extreme multi-label classification with Omikuji 🦀**

Scikit-Omikuji is a Python package that provides a scikit-learn compatible interface to the high-performance [Omikuji](https://github.com/tomtung/omikuji) library for extreme multi-label classification. Built on top of the Rust-based Omikuji implementation, it offers significant performance improvements while maintaining ease of use through familiar scikit-learn APIs.

## 🚀 Key Features

- **🔥 High Performance**: Built on Rust-based Omikuji implementation, 1.3x to 4.6x faster than existing implementations
- **🧬 Scikit-learn Compatible**: Drop-in replacement for scikit-learn classifiers with familiar `fit`, `predict`, and `predict_proba` methods
- **💾 Memory Efficient**: Direct support for `scipy.sparse.csr_matrix` without expensive I/O operations
- **🌳 Advanced Algorithms**: Implementation of PARABEL (Partitioned Label Trees) and its variations
- **⚡ Parallel Processing**: Multi-threaded training and prediction with automatic CPU utilization
- **🎛️ Flexible Configuration**: Extensive hyperparameter tuning options and YAML-based configuration
- **📊 Rich Metrics**: Comprehensive evaluation metrics for extreme multi-label classification
- **🖥️ CLI Interface**: Command-line tools for training and evaluation
- **📈 Scalable**: Handles datasets with millions of labels efficiently

## 📦 Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/scikit-omikuji.git
cd scikit-omikuji

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Prerequisites

- **Python**: 3.10 or higher
- **Rust**: Required for building the underlying Omikuji library
- **System Dependencies**:
  - On Ubuntu/Debian: `sudo apt-get install build-essential`
  - On macOS: Install Xcode command line tools
  - On Windows: Install Visual Studio Build Tools

## 🚀 Quick Start

```python
import numpy as np
from scipy.sparse import csr_array
from sklearn.datasets import make_multilabel_classification
from skomikuji import OmikujiClassifier
from skomikuji.metrics import compute_metrics

# Generate synthetic multilabel dataset
X_train, y_train = make_multilabel_classification(
    n_samples=1000, n_features=100, n_labels=10, n_classes=50,
    sparse=True, random_state=42, allow_unlabeled=False
)

X_test, y_test = make_multilabel_classification(
    n_samples=200, n_features=100, n_labels=10, n_classes=50,
    sparse=True, random_state=43, allow_unlabeled=False
)

# Convert to required dtypes
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.uint32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.uint32)

# Train the classifier
classifier = OmikujiClassifier(
    n_trees=3,
    max_depth=20,
    linear_c=1.0,
    top_k=10
)
classifier.fit(X_train, y_train)

# Make predictions
y_pred_proba = classifier.predict_proba(X_test)
y_pred = classifier.predict(X_test, proba_threshold=0.5)

# Evaluate performance
metrics = compute_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_score=y_pred_proba,
    k=5
)
print(metrics)
```

## 📖 Usage Guide

### Basic Classification

```python
from skomikuji import OmikujiClassifier

# Initialize with custom parameters
classifier = OmikujiClassifier(
    n_trees=5,              # Number of trees in the ensemble
    max_depth=30,           # Maximum tree depth
    linear_c=2.0,           # Regularization parameter
    cluster_k=4,            # Number of clusters for k-means
    beam_size=20,           # Beam size for prediction
    loss_type="log",        # Loss function: "hinge" or "log"
    n_jobs=-1               # Use all available CPU cores
)

# Fit the model
classifier.fit(X_train, y_train)

# Predict probabilities
probabilities = classifier.predict_proba(X_test)

# Predict binary labels with custom threshold
predictions = classifier.predict(X_test, proba_threshold=0.3)
```

### Advanced Configuration

```python
# Using YAML configuration
import yaml
from skomikuji import OmikujiClassifier

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

classifier = OmikujiClassifier(**config['classifier_params'])
classifier.fit(X_train, y_train)
```

### Evaluation Metrics

```python
from skomikuji.metrics import (
    precision_at_k, recall_at_k, f1_at_k, ndcg_at_k,
    compute_metrics
)

# Individual metrics
p_at_5 = precision_at_k(y_true, y_pred, k=5)
r_at_5 = recall_at_k(y_true, y_pred, k=5)
f1_at_5 = f1_at_k(y_true, y_pred, k=5)
ndcg_at_5 = ndcg_at_k(y_true, y_score, k=5)

# Comprehensive evaluation
all_metrics = compute_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_score=y_pred_proba,
    k=5,
    propensity_coeff=(0.5, 0.4)  # For propensity-scored metrics
)
```

## 🎯 API Reference

### OmikujiClassifier

The main classifier class with scikit-learn compatible interface.

#### Parameters

- **`n_trees`** (int, default=3): Number of trees in the ensemble
- **`max_depth`** (int, default=20): Maximum depth of each tree
- **`min_branch_size`** (int, default=100): Minimum samples for branching
- **`linear_c`** (float, default=1.0): Regularization parameter for linear classifiers
- **`cluster_k`** (int, default=2): Number of clusters for k-means
- **`beam_size`** (int, default=10): Beam size for beam search during prediction
- **`loss_type`** (str, default="hinge"): Loss function ("hinge" or "log")
- **`n_jobs`** (int, default=None): Number of parallel jobs

#### Methods

- **`fit(X, y)`**: Train the classifier
- **`predict_proba(X)`**: Predict class probabilities
- **`predict(X, proba_threshold=0.5)`**: Predict binary labels

### Grid Search

```bash
skomikuji train -i /path/to/data -g --config_file config.yaml
```

## ⚙️ Configuration

Scikit-Omikuji supports YAML-based configuration for reproducible experiments:

```yaml
# config.yaml
classifier_params:
  n_trees: 5
  max_depth: 30
  linear_c: 2.0
  cluster_k: 4
  beam_size: 20
  loss_type: "log"
  n_jobs: -1

grid_search:
  n_jobs: -1
  verbose: 1
  cross_validation:
    n_folds: 5
    random_state: 42
  param_grid:
    n_trees: [3, 5, 7]
    max_depth: [20, 30, 40]
    linear_c: [0.5, 1.0, 2.0]
```

## 📊 Performance Benchmarks

Scikit-Omikuji maintains the performance advantages of the original Omikuji implementation:

| Dataset         | Metric | Original Parabel | Scikit-Omikuji | Speedup |
|-----------------|--------|------------------|----------------|---------|
| EURLex-4K       | P@1    | 82.2            | 82.1           | 1.3x    |
|                 | P@3    | 68.8            | 68.8           | faster  |
|                 | P@5    | 57.6            | 57.7           |         |
| Amazon-670K     | P@1    | 44.9            | 44.8           | 1.7x    |
|                 | P@3    | 39.8            | 39.8           | faster  |
|                 | P@5    | 36.0            | 36.0           |         |
| WikiLSHTC-325K  | P@1    | 65.0            | 64.8           | 1.5x    |
|                 | P@3    | 43.2            | 43.1           | faster  |
|                 | P@5    | 32.0            | 32.1           |         |

*Benchmarks run on Intel® Core™ i7-6700 CPU (4 cores)*

## 🔧 Data Requirements

### Critical Data Type Requirements

Scikit-Omikuji uses efficient memory mapping between Python and Rust, requiring specific data types:

- **Features**: Must be `np.float32` dtype
- **Labels**: Must be `np.uint32` dtype
- **Format**: Sparse matrices (`scipy.sparse.csr_matrix` or `csr_array`) preferred

```python
# Correct data preparation
X = X.astype(np.float32)  # Features as float32
y = y.astype(np.uint32)   # Labels as uint32

# Incorrect - will raise TypeError
X = X.astype(np.float64)  # ❌ Wrong dtype
y = y.astype(np.int32)    # ❌ Wrong dtype
```

### Sparse Matrix Support

```python
from scipy.sparse import csr_array

# Direct sparse matrix support (recommended)
X_sparse = csr_array(X, dtype=np.float32)
y_sparse = csr_array(y, dtype=np.uint32)

classifier.fit(X_sparse, y_sparse)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-username/scikit-omikuji.git
cd scikit-omikuji
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black src/ tests/
ruff check src/ tests/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html
```

## ❓ FAQ

### Q: Why do I get a TypeError about data types?

**A:** Scikit-Omikuji requires specific data types for efficient memory mapping:

- Features must be `np.float32`
- Labels must be `np.uint32`

```python
# Fix dtype issues
X = X.astype(np.float32)
y = y.astype(np.uint32)
```

### Q: How do I handle large datasets?

**A:** Use sparse matrices and adjust memory-related parameters:

```python
from scipy.sparse import csr_array

# Use sparse matrices
X_sparse = csr_array(X, dtype=np.float32)

# Adjust parameters for memory efficiency
classifier = OmikujiClassifier(
    train_trees_1_by_1=True,  # Train trees sequentially to save memory
    n_jobs=1                   # Reduce parallelism if memory constrained
)
```

### Q: How do I tune hyperparameters?

**A:** Use the built-in grid search functionality or scikit-learn's tools:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_trees': [3, 5, 7],
    'max_depth': [20, 30, 40],
    'linear_c': [0.5, 1.0, 2.0]
}

grid_search = GridSearchCV(
    OmikujiClassifier(),
    param_grid,
    cv=3,
    scoring='f1_macro'
)
grid_search.fit(X_train, y_train)
```

### Q: Can I use this with scikit-learn pipelines?

**A:** Yes! OmikujiClassifier is fully compatible with scikit-learn:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', OmikujiClassifier())
])
pipeline.fit(X_train, y_train)
```

## 🔍 Troubleshooting

### Installation Issues

**Rust Compilation Errors:**

```bash
# Install Rust if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Update Rust
rustup update
```

**Missing System Dependencies:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

### Runtime Issues

**Memory Errors:**

- Reduce `n_jobs` parameter
- Set `train_trees_1_by_1=True`
- Use smaller `beam_size`

**Slow Training:**

- Increase `n_jobs` (up to number of CPU cores)
- Reduce `max_depth` or `n_trees`
- Increase `min_branch_size`

**Poor Performance:**

- Tune `linear_c` parameter
- Adjust `cluster_k` for your dataset
- Try different `loss_type` ("hinge" vs "log")

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

If you use Scikit-Omikuji in your research, please cite the original papers:

```bibtex
@inproceedings{prabhu2018parabel,
  title={Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising},
  author={Prabhu, Yashoteja and Kag, Anil and Harsola, Shrutendra and Agrawal, Rahul and Varma, Manik},
  booktitle={Proceedings of the 2018 World Wide Web Conference},
  pages={993--1002},
  year={2018}
}

@article{khandagale2019bonsai,
  title={Bonsai-Diverse and Shallow Trees for Extreme Multi-label Classification},
  author={Khandagale, Sujay and Xiao, Han and Babbar, Rohit},
  journal={arXiv preprint arXiv:1904.08249},
  year={2019}
}
```

## 🙏 Acknowledgments

- **[Tom Tung](https://github.com/tomtung)** for the original [Omikuji](https://github.com/tomtung/omikuji) implementation
- The Rust community for excellent performance-focused libraries
- The scikit-learn team for the excellent API design patterns
- Contributors to the extreme multi-label classification research community

## 📞 Support

- **Documentation**: [Read the Docs](https://scikit-omikuji.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/your-username/scikit-omikuji/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/scikit-omikuji/discussions)

---

<p align="center">
  <strong>🏷️ Efficient Extreme Multi-Label Classification with Scikit-Omikuji 🦀</strong>
</p>
