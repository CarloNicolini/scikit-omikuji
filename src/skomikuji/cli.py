import click
import joblib
from typing import Optional, List
from pathlib import Path
import numpy as np
from scipy.sparse import csr_array
import pandas as pd
import json
import rich
from tabulate import tabulate
from skomikuji.metrics import psprecision_at_k_scorer, compute_metrics
from skomikuji import OmikujiClassifier
from sklearn.model_selection import GridSearchCV,  KFold


def print_dictionary(dictionary):
    # Dumping dictionary as JSON with color formatting
    rich.print_json(json.dumps(dictionary, indent=4))


def load_parquets(
    input_path: str | Path, folds_list: List[str], to_sparse: bool = False
):
    """
    Load the given list of parquet files
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


@click.group(
    name="cli", help="Base command line interface for the smartscreening python package"
)
def cli():
    pass


@click.command()
@click.option("-i", "--input-path", type=click.Path(exists=True), required=True)
@click.option("-k", type=click.INT, required=False, default=1)
@click.option("-a", "--propensity_a", type=float, required=False, default=0.65)
@click.option("-b", "--propensity_b", type=float, required=False, default=2.8)
@click.option("-p", "--print_result", type=bool, required=False, default=True)
def predict(
    input_path: Path, k: int = 1, propensity_a=0.65, propensity_b=2.8, print_result=True
):
    return _predict_from_path(input_path, k, propensity_a, propensity_b, print_result)


def _predict_from_path(
    input_path: Path,
    k: int = 1,
    propensity_a=0.65,
    propensity_b=2.8,
    print_result=False,
):
    """
    Compute the model predictions metrics

    Parameters
    ----------

    path: Path
        The path to extract
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


@click.command()
@click.option("-i", "--input-path", type=click.Path(exists=True), required=True)
@click.option("-k", type=click.INT, required=False, default=1)
@click.option("-a", "--propensity_a", type=float, required=False, default=0.65)
@click.option("-b", "--propensity_b", type=float, required=False, default=2.8)
@click.option("-g", "--grid-search", type=bool, required=False)
@click.option("-c", "--config_file", type=click.Path(exists=True), required=False)
def train(
    input_path: Path,
    k: int = 1,
    propensity_a=0.65,
    propensity_b=2.8,
    grid_search: bool=False,
    config_file: Optional[Path] = None,
):
    model = OmikujiClassifier()
    params={}
    if isinstance(config_file, (str,Path)):
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
    pd.Series(model.get_params()).to_json(Path(input_path)/"omikuji_params.json")
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
    comparison.reset_index().to_csv(Path(input_path) / "test_results_omikuji.csv",index=False)
    print(tabulate(comparison, headers="keys", tablefmt="pretty", floatfmt=".3f"))

cli.add_command(train)
cli.add_command(predict)


def main():
    cli()


if __name__ == "__main__":
    main()
