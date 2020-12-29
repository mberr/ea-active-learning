"""Collate results from MLFlow and generate artifacts."""
import argparse
import logging
from typing import Optional

import pandas

from kgm.eval.active_learning import aggregate_auc_qh, auc_qh, get_significance, metrics_long_to_wide
from kgm.utils.mlflow_utils import get_results


def _auc_table(metrics: pandas.DataFrame, params: pandas.DataFrame):
    """Create the AUC table."""
    # compute auc
    auc = auc_qh(metrics=metrics, params=params)
    # combine with parameters
    auc = params.merge(right=auc, how="inner", on="run_id")
    # aggregate auc
    auc = aggregate_auc_qh(auc=auc)
    # significance
    auc = get_significance(auc=auc)
    # prepare table
    auc["auc"] = (
        "$"
        + auc[("auc_qh", "mean")].apply("{:.4f}".format)
        + r" \pm "
        + auc[("auc_qh", "std")].apply("{:.4f}".format)
        + "$"
        + auc[("auc_qh", "significance")].apply(lambda s: "*" if s else "")
    )
    auc = auc[["heuristic.heuristic_name", "data.subset_name", "auc"]].copy()
    auc.columns = ["heuristic", "subset", "AUC H@1"]
    auc = auc.set_index(["heuristic", "subset"]).unstack()
    print(auc.to_latex(escape=False))


def _plots(
    metrics: pandas.DataFrame,
    params: pandas.DataFrame,
    num_queries_column: str = "queries.num_queries",
    hits_column: str = "evaluation.test.hits_at_1",
    subset_column: str = "data.subset_name",
    heuristic_column: str = "heuristic.heuristic_name",
    dropout_column: str = "model.node_embedding_dropout",
    max_queries: Optional[int] = None,
):
    """Create plots."""
    try:
        from matplotlib import pyplot as plt
        import seaborn
    except ImportError:
        raise RuntimeError("To generate plots please install matplotlib and seaborn using \n\n\tpip install -U seaborn matplotlib")
    # num_queries, hits
    metrics = metrics_long_to_wide(metrics=metrics, num_queries_column=num_queries_column, hits_column=hits_column)[["run_id", "step", num_queries_column, hits_column]]
    # subset, heuristic, dropout
    params = params[["run_id", subset_column, heuristic_column, dropout_column]]
    data = metrics.merge(right=params, on="run_id", how="inner").set_index(["run_id", "step"])
    # consistent ordering
    data = data[[subset_column, heuristic_column, dropout_column, num_queries_column, hits_column]]
    # rename
    data.columns = ["subset", "heuristic", "dropout", "num_queries", "Test H@1"]
    grid = seaborn.relplot(
        data=data,
        x="num_queries",
        y="Test H@1",
        hue="heuristic",
        col="subset",
        col_order=["en_de", "en_fr"],
        style="dropout",
        kind="line",
        ci="sd",
        facet_kws=dict(sharex=False),
    )
    grid.set(xlim=(0, max_queries), ylim=(0, None))
    plt.savefig("plot.pdf" if max_queries is None else "plot_first.pdf")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000')
    parser.add_argument('--force', action="store_true")
    args = parser.parse_args()

    params, metrics = get_results(
        tracking_uri=args.tracking_uri,
        experiment_name="active_learning",
        force=args.force,
    )

    # normalize heuristic_name
    translation = {
        "BALD": "bald",
        "MostProbableMatchingInUnexploredRegion": "esccn",
        "PreviousExperienceBased": "prexp",
        "approximatevertexcover": "avc",
        "betweenness": "betw",
        "coreset": "cs",
        "degree": "deg",
        "random": "rnd",
    }
    params["heuristic.heuristic_name"] = params["heuristic.heuristic_name"].apply(translation.__getitem__)

    _auc_table(metrics=metrics, params=params)
    _plots(metrics=metrics, params=params)
    _plots(metrics=metrics, params=params, max_queries=2_000)
