import json
from pathlib import Path
from typing import Annotated, Optional

import joblib
import numpy as np
import pandas as pd
import rich
import typer
from scipy.sparse import csr_array
from sklearn.model_selection import GridSearchCV, KFold
from tabulate import tabulate

from skomikuji import OmikujiClassifier
from skomikuji.metrics import compute_metrics, psprecision_at_k_scorer


def print_dictionary(dictionary):
    # Dumping dictionary as JSON with color formatting
    rich.print_json(json.dumps(dictionary, indent=4))


def load_parquets(
    input_path: str | Path, folds_list: list[str], to_sparse: bool = False
):
    """
    Load the given list of parquet files.

    Parameters
    ----------
    input_path : str | Path
        Path to the directory containing parquet files
    folds_list : list[str]
        List of parquet file names (without extension) to load
    to_sparse : bool, default=False
        Whether to convert loaded data to sparse format

    Returns
    -------
    dict
        Dictionary containing loaded data with fold names as keys
    """
    data = {}
    for fold in folds_list:
        data[fold] = pd.read_parquet(f"{input_path}/{fold}.parquet")
        if fold[0] == "X":
            data[fold] = data[fold].astype(np.float32)
        elif fold[0] == "Y" and "proba" not in fold:
            data[fold] = data[fold].astype(np.uint32)
        elif fold[0] == "Y" and "proba" in fold:
            data[fold] = data[fold].astype(np.float32)
        if to_sparse:
            data[fold] = csr_array(data[fold])
    return data


# Create the main Typer app
app = typer.Typer(
    name="skomikuji-cli",
    help="Command line interface for the scikit-omikuji package - multilabel classification in extreme settings.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def predict(
    input_path: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input-path",
            help="Path to directory containing prediction data",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    k: Annotated[
        int,
        typer.Option("-k", help="Number of top predictions to consider for metrics"),
    ] = 1,
    propensity_a: Annotated[
        float, typer.Option("-a", "--propensity-a", help="Propensity score parameter A")
    ] = 0.65,
    propensity_b: Annotated[
        float, typer.Option("-b", "--propensity-b", help="Propensity score parameter B")
    ] = 2.8,
    print_result: Annotated[
        bool,
        typer.Option(
            "-p", "--print-result", help="Whether to print results to console"
        ),
    ] = True,
):
    """
    Compute model prediction metrics from saved test data.

    This command loads test predictions and computes various multilabel classification metrics
    including precision, recall, F1-score, and propensity-scored metrics.
    """
    return _predict_from_path(input_path, k, propensity_a, propensity_b, print_result)


def _predict_from_path(
    input_path: Path,
    k: int = 1,
    propensity_a: float = 0.65,
    propensity_b: float = 2.8,
    print_result: bool = False,
):
    """
    Compute model prediction metrics from saved test data.

    Parameters
    ----------
    input_path : Path
        Path to directory containing parquet files with test data
    k : int, default=1
        Number of top predictions to consider for metrics
    propensity_a : float, default=0.65
        Propensity score parameter A
    propensity_b : float, default=2.8
        Propensity score parameter B
    print_result : bool, default=False
        Whether to print results to console

    Returns
    -------
    dict
        Dictionary containing computed metrics
    """
    data = load_parquets(
        input_path=input_path,
        folds_list=["Y_test", "Y_test_proba_pred", "Y_test_pred"],
        to_sparse=True,
    )

    metrics = compute_metrics(
        y_true=data["Y_test"],
        y_pred=data["Y_test_pred"],
        y_score=data["Y_test_proba_pred"],
        k=k,
        propensity_coeff=(propensity_a, propensity_b),
    )
    if print_result:
        print_dictionary(metrics)
    return metrics


def train(
    input_path: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input-path",
            help="Path to directory containing training data",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    k: Annotated[
        int,
        typer.Option("-k", help="Number of top predictions to consider for metrics"),
    ] = 1,
    propensity_a: Annotated[
        float, typer.Option("-a", "--propensity-a", help="Propensity score parameter A")
    ] = 0.65,
    propensity_b: Annotated[
        float, typer.Option("-b", "--propensity-b", help="Propensity score parameter B")
    ] = 2.8,
    grid_search: Annotated[
        bool,
        typer.Option(
            "-g", "--grid-search", help="Enable grid search for hyperparameter tuning"
        ),
    ] = False,
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "-c",
            "--config-file",
            help="Path to configuration file (YAML or JSON)",
            exists=True,
        ),
    ] = None,
):
    """
    Train an Omikuji classifier on the provided dataset.

    This command trains a multilabel classifier using the Omikuji algorithm, with optional
    grid search for hyperparameter optimization. Training data should be provided as
    parquet files containing features (X_train, X_test) and labels (Y_train, Y_test).

    The trained model will be evaluated against test data and compared with baseline metrics
    if available. Results are saved to the input directory.
    """
    model = OmikujiClassifier()
    params = {}
    if isinstance(config_file, (str, Path)):
        config_file = Path(config_file)
        if config_file.exists() and config_file.suffix == ".yaml":
            try:
                from yaml import safe_load
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please install the PyYaml package to continue."
                )
            params = safe_load(open(config_file, "r"))
        elif config_file.exists() and config_file.suffix == ".json":
            import json

            params = json.loads(config_file.read_text())
        else:
            raise FileNotFoundError(
                f"The {config_file} does not exist or has not the correct format {config_file.suffix}"
            )
    if grid_search:
        print_dictionary(params["grid_search"])

        model = GridSearchCV(
            estimator=OmikujiClassifier(n_jobs=1),
            param_grid=params["grid_search"]["param_grid"],
            refit=params["grid_search"]["cross_validation"]["refit_scorer"],
            scoring=psprecision_at_k_scorer,
            cv=KFold(
                params["grid_search"]["cross_validation"]["n_folds"],
                shuffle=True,
                random_state=params["grid_search"]["cross_validation"]["random_state"],
            ),
            n_jobs=params["grid_search"]["n_jobs"],
            verbose=params["grid_search"]["verbose"],
            error_score="raise",
        )
    else:
        model.set_params(**params["classifier_params"])

    data = load_parquets(
        input_path=input_path,
        folds_list=["Y_test", "Y_train", "X_train", "X_test"],
        to_sparse=True,
    )

    model.fit(data["X_train"], data["Y_train"])

    print_dictionary(model.get_params())
    pd.Series(model.get_params()).to_json(Path(input_path) / "omikuji_params.json")
    data["Y_test_proba_pred"] = model.predict_proba(data["X_test"])
    data["Y_test_pred"] = (data["Y_test_proba_pred"] > 0.5).astype(np.uint32)

    metrics = compute_metrics(
        y_true=data["Y_test"],
        y_pred=data["Y_test_pred"],
        y_score=data["Y_test_proba_pred"],
        k=k,
        propensity_coeff=(propensity_a, propensity_b),
    )

    base_metrics = pd.Series(
        _predict_from_path(
            input_path, k, propensity_a, propensity_b, print_result=False
        )
    ).rename("XGBoost")
    omikuji_metrics = pd.Series(metrics).rename("Omikuji")

    comparison = pd.concat([base_metrics, omikuji_metrics], axis=1)
    comparison.reset_index().to_csv(
        Path(input_path) / "test_results_omikuji.csv", index=False
    )
    print(tabulate(comparison, headers="keys", tablefmt="pretty", floatfmt=".3f"))


# Register commands with the app
app.command()(predict)
app.command()(train)


def main():
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
