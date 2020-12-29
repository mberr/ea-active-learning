"""Evaluation utilities for active learning heuristics."""
import pandas
import scipy.stats
import sklearn.metrics


def auc_qh(
    metrics: pandas.DataFrame,
    params: pandas.DataFrame,
    num_queries_column: str = "queries.num_queries",
    hits_column: str = "evaluation.test.hits_at_1",
    subset_column: str = "data.subset_name"
) -> pandas.DataFrame:
    """
    Compute the AUC for the number of queries vs. hits at 1 curve.

    :param metrics:
        A dataframe of metrics with at least the following columns: {"run_id", "step", "key", "value"}, where there are
        key-value pairs for key=num_queries_column and key=hits_column.
    :param params:
        A dataframe of parameters with at leas the following columns: {"run_id", subset_column}
    :param num_queries_column:
        The name of the key for number of queries.
    :param hits_column:
        The name of the key for hits@1.
    :param subset_column:
        The name of the column for subset.

    :return:
        A dataframe with columns {"run_id", "auc_qh"}
    """
    metrics = metrics_long_to_wide(metrics=metrics, num_queries_column=num_queries_column, hits_column=hits_column)
    metrics = metrics.merge(right=params[["run_id", subset_column]], how="inner", on="run_id")

    auc = []
    for subset, subset_group in metrics.groupby(by=subset_column):
        # get largest common step
        max_num_queries = int(subset_group.groupby(by="run_id").agg({num_queries_column: "max"}).min())
        subset_group = subset_group[subset_group[num_queries_column] <= max_num_queries]
        for run_id, run_id_group in subset_group.groupby(by="run_id"):
            # sklearn.metrics.auc expects x to be sorted
            run_id_group = run_id_group.sort_values(by=num_queries_column)
            x = run_id_group[num_queries_column] / max_num_queries
            y = run_id_group[hits_column]
            this_auc = sklearn.metrics.auc(x, y)
            auc.append((run_id, this_auc))
    return pandas.DataFrame(data=auc, columns=["run_id", "auc_qh"])


def metrics_long_to_wide(
    metrics: pandas.DataFrame,
    num_queries_column: str,
    hits_column: str,
) -> pandas.DataFrame:
    """
    Convert a dataframe with metrics from long format to wide format.

    :param metrics:
        The dataframe of metrics. Has at least columns {"run_id", "step", "key", "value"}.
    :param num_queries_column:
        The name of the number of queries key.
    :param hits_column:
        The name of the hits@1 key.

    :return:
        A dataframe in wide format with columns {"run_id", "step", num_queries_column, hits_column}
    """
    metrics = metrics[metrics["key"].isin([num_queries_column, hits_column])]
    metrics = metrics.pivot(index=["run_id", "step"], columns="key", values="value")
    return metrics.reset_index()


def aggregate_auc_qh(
    auc: pandas.DataFrame,
    subset_column: str = "data.subset_name",
    heuristic_column: str = "heuristic.heuristic_name",
    auc_column: str = "auc_qh",
) -> pandas.DataFrame:
    """
    Aggregate AUC QH.

    :param auc:
        The dataframe containing AUC-QH values for each run. Has at least the following columns: 
        {heuristic_column, auc_column}.
    :param subset_column:
        The name of the subset column.
    :param heuristic_column:
        The name of the heuristic column.
    :param auc_column:
        The name of the AUC-QH column.

    :return:
        A dataframe with columns
        {
            ("", subset_column),
            ("", heuristic_column),
            (auc_column, "mean"),
            (auc_column, "std"),
            (auc_column, "count"),
        }
    """
    return auc.groupby(by=[subset_column, heuristic_column]).agg({auc_column: ["mean", "std", "count"]}).reset_index()


def get_significance(
    auc: pandas.DataFrame,
    baseline: str = "rnd",
    threshold: float = 0.01,
    equal_var: bool = False,
    subset_column: str = "data.subset_name",
    heuristic_column: str = "heuristic.heuristic_name",
    auc_column: str = "auc_qh",
) -> pandas.DataFrame:
    """
    Compute significance of results against a baseline using Welch's t-test.

    :param auc: A dataframe with columns
        {
            ("", subset_column),
            ("", heuristic_column),
            (auc_column, "mean"),
            (auc_column, "std"),
            (auc_column, "count")
        }
    :param baseline:
        The baseline heuristic.
    :param threshold:
        The significance threshold.
    :param equal_var:
        Whether to assume equal variance. If False, us Welch's t-test. Otherwise, use default t-test.
    :param subset_column:
        The name of the subset column.
    :param heuristic_column:
        The name of the heuristic column.
    :param auc_column:
        The name of the AUC-QH column.

    :return:
        A dataframe with columns
        {
            ("", subset_column),
            ("", heuristic_column),
            (auc_column, "mean"),
            (auc_column, "std"),
            (auc_column, "count"),
            (auc_column, "significance"),
        }
    """
    sig_data = []
    for subset_name, group in auc.groupby(by=(subset_column, "")):
        columns = [(auc_column, "mean"), (auc_column, "std"), (auc_column, "count")]
        b_mean, b_std, b_num = group.loc[group[heuristic_column] == baseline, columns].iloc[0].values
        for heuristic, subgroup in group.groupby(by=(heuristic_column, "")):
            h_mean, h_std, h_num = subgroup[columns].iloc[0].values
            _, p = scipy.stats.ttest_ind_from_stats(b_mean, b_std, b_num, h_mean, h_std, h_num, equal_var=equal_var)
            significant = p < threshold
            sig_data.append((subset_name, heuristic, significant))
    significance = pandas.DataFrame(data=sig_data, columns=[(subset_column, ""), (heuristic_column, ""), (auc_column, "significance")])
    return auc.merge(right=significance, how="inner", on=[(subset_column, ""), (heuristic_column, "")])
