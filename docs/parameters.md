# Parameters

The OmikujiClassifier has several parameters that can be adjusted to optimize its performance. Here's a detailed description of each parameter:

## Main Parameters

- `top_k` (int, default=5): The number of labels to return in prediction.
- `beam_size` (int, default=10): The maximum number of paths that can be traversed in a tree.
- `n_trees` (int, default=3): The number of trees in the ensemble.
- `min_branch_size` (int, default=100): The number of labels below which no further clustering & branching is done.
- `max_depth` (int, default=20): The maximum tree depth.

## Clustering Parameters

- `cluster_k` (int, default=2): The number of clusters.
- `cluster_balanced` (bool, default=False): Whether to perform balanced k-means clustering instead of regular k-means clustering.
- `cluster_eps` (float, default=0.0001): The epsilon value for determining linear classifier convergence.
- `cluster_min_size` (int, default=2): Labels in clusters with sizes smaller than this threshold are reassigned to other clusters.

## Linear Classifier Parameters

- `linear_eps` (float, default=0.1): The epsilon value for determining linear classifier convergence.
- `linear_c` (float, default=1): The cost coefficient for regularizing linear classifiers.
- `linear_weight_threshold` (float, default=0.1): The threshold for pruning weight vectors of linear classifiers.
- `linear_max_iter` (int, default=20): The maximum number of iterations for training each linear classifier.
- `loss_type` (str, default="hinge"): The loss function used by linear classifiers. Can be "hinge" or "log".

## Other Parameters

- `centroid_threshold` (float, default=0.0): The threshold for pruning label centroid vectors.
- `collapse_every_n_layers` (int, default=0): The number of adjacent layers to collapse. This increases tree arity and decreases tree depth.
- `n_jobs` (int, optional): The number of worker threads. If 0 or None, the number is selected automatically.
- `check_no_labels` (bool, default=False): Whether to check in the fit method if some samples have no labels.

For more details on how these parameters affect the model's performance, refer to the [Parabel algorithm paper](http://manikvarma.org/pubs/prabhu18b.pdf).