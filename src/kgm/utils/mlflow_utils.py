# coding=utf-8
"""Utilities for MLflow."""
import hashlib
import itertools
import logging
import math
import os
import pathlib
import platform
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import mlflow
import mlflow.entities
import pandas
from tqdm.auto import tqdm

from kgm.utils.common import to_dot

logger = logging.getLogger(__name__)


def log_params_to_mlflow(
    config: Dict[str, Any],
    prefix: Optional[str] = None,
) -> None:
    """Log parameters to MLFlow. Allows nested dictionaries."""
    nice_config = to_dot(config, prefix=prefix)
    # mlflow can only process 100 parameters at once
    keys = sorted(nice_config.keys())
    batch_size = 100
    for start in range(0, len(keys), batch_size):
        mlflow.log_params({k: nice_config[k] for k in keys[start:start + batch_size]})


def log_metrics_to_mlflow(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """Log metrics to MLFlow. Allows nested dictionaries."""
    nice_metrics = to_dot(metrics, prefix=prefix)
    nice_metrics = {k: float(v) for k, v in nice_metrics.items()}
    mlflow.log_metrics(nice_metrics, step=step)


def run_experiments(
    search_list: List[Mapping[str, Any]],
    experiment: Callable[[Mapping[str, Any]], Tuple[Mapping[str, Any], int]],
    num_replicates: int = 1,
    break_on_error: bool = False,
) -> None:
    """
    Run experiments synchronized by MLFlow.

    :param search_list:
        The search list of parameters. Each entry corresponds to one experiment.
    :param experiment:
        The experiment as callable. Takes the dictionary of parameters as input, and produces a result dictionary as well as a final step.

    :return: None
    """

    # randomize sort order to avoid collisions with multiple workers
    def key(x: Mapping[str, Any]):
        return hashlib.md5((';'.join(f'{k}={x}' for k, v in x.items()) + ';' + str(platform.node()) + ';' + str(os.getenv('CUDA_VISIBLE_DEVICES', '?'))).encode()).hexdigest()

    search_list = sorted(search_list, key=key)

    n_experiments = len(search_list)
    counter = {
        'error': 0,
        'success': 0,
        'skip': 0,
    }
    for run, params in enumerate(search_list * num_replicates):
        logging.info(f'================== Run {run}/{n_experiments * num_replicates} ==================')
        params = dict(**params)

        # Check, if run with current parameters already exists
        query = ' and '.join(list(map(lambda item: f"params.{item[0]} = '{str(item[1])}'", to_dot(params).items())))
        logging.info(f'Query: \n{query}\n')

        run_hash = hashlib.md5(query.encode()).hexdigest()
        params['run_hash'] = run_hash
        logging.info(f'Hash: {run_hash}')

        existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{run_hash}'", run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY)
        if len(existing_runs) >= num_replicates:
            logging.info('Skipping existing run.')
            counter['skip'] += 1
            continue

        mlflow.start_run()

        params['environment'] = {
            'server': platform.node(),
        }

        # Log to MLFlow
        log_params_to_mlflow(params)
        log_metrics_to_mlflow({'finished': False}, step=0)

        # Run experiment
        try:
            final_evaluation, final_step = experiment(params)
            # Log to MLFlow
            log_metrics_to_mlflow(metrics=final_evaluation, step=final_step)
            log_metrics_to_mlflow({'finished': True}, step=final_step)
            counter['success'] += 1
        except Exception as e:
            logging.error('Error occured.')
            logging.exception(e)
            log_metrics_to_mlflow(metrics={'error': 1})
            counter['error'] += 1
            if break_on_error:
                raise e

        mlflow.end_run()

    logging.info(f'Ran {counter} experiments.')


def _has_non_empty_metrics(
    run: mlflow.entities.Run,
) -> bool:
    """Return true if the run has at least one finite metric value."""
    metrics = run.data.metrics
    return len(metrics) > 0 and any(map(math.isfinite, metrics.values()))


T = TypeVar('T')


def _get_run_information_from_experiments(
    *,
    projection: Callable[[mlflow.entities.Run], T],
    experiment_ids: Union[int, Collection[int]],
    selection: Optional[Callable[[mlflow.entities.Run], bool]] = None,
    filter_string: Optional[str] = "",
    tracking_uri: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> Collection[T]:
    """
    Collect information for all runs associated with an experiment ID.

    .. note ::
        Exactly one of `tracking_uri` or `client` has to be provided.

    :param projection:
        The projection from an MLFlow run to the desired information.
    :param experiment_ids:
        The experiment IDs.
    :param selection:
        A selection criterion for filtering runs.
    :param filter_string:
        Filter query string, defaults to searching all runs.
    :param tracking_uri:
        The Mlflow tracking URI.
    :param client:
        The Mlflow client.


    :return:
        A collection of information for each run.
    """
    # Normalize input
    if not isinstance(experiment_ids, (list, tuple)):
        experiment_ids = [experiment_ids]
    experiment_ids = list(experiment_ids)
    if None not in {tracking_uri, client}:
        raise ValueError('Cannot provide tracking_uri and client.')
    if tracking_uri is not None:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    if selection is None:
        def selection(_run: mlflow.entities.Run) -> bool:
            """Keep all."""
            return True

    runs = []

    # support for paginated results
    continue_searching = True
    page_token = None

    while continue_searching:
        page_result_list = client.search_runs(
            experiment_ids=list(map(str, experiment_ids)),
            filter_string=filter_string,
            page_token=page_token,
        )
        runs.extend(projection(run) for run in page_result_list if selection(run))
        page_token = page_result_list.token
        continue_searching = page_token is not None

    return runs


def get_params_from_experiments(
    *,
    experiment_ids: Union[int, Collection[int]],
    filter_string: str = "",
    selection: Optional[Callable[[mlflow.entities.Run], bool]] = None,
    tracking_uri: Optional[str] = None,
    client: Optional[mlflow.tracking.MlflowClient] = None,
) -> pandas.DataFrame:
    """
    Collect run parameters for all runs associated with an experiment ID.

    .. note ::
        Exactly one of `tracking_uri` or `client` has to be provided.

    :param experiment_ids:
        The experiment IDs.
    :param filter_string:
        A filter for runs.
    :param selection:
        A selection criterion for filtering runs.
    :param tracking_uri:
        The Mlflow tracking URI.
    :param client:
        The Mlflow client.


    :return:
        A dataframe with an index of `run_uuid`s and one column per parameter.
    """

    def _get_run_params(
        run: mlflow.entities.Run,
    ) -> Mapping[str, str]:
        """Extract the run parameters."""
        result = dict(run.data.params)
        result['run_id'] = run.info.run_uuid
        return result

    return pandas.DataFrame(data=_get_run_information_from_experiments(
        projection=_get_run_params,
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        selection=selection,
        tracking_uri=tracking_uri,
        client=client,
    )).set_index(keys='run_id')


def buffered_load_parameters(
    tracking_uri: str,
    exp_id: int,
    buffer_root: pathlib.Path,
    force: bool = False,
    filter_string: str = "",
) -> pandas.DataFrame:
    """
    Load parameters from MLflow with TSV-file buffering.

    :param tracking_uri:
        The tracking URI.
    :param exp_id:
        The experiment ID.
    :param buffer_root:
        The root directory for buffering. Must be an existing directory.
    :param force:
        Whether to enforce re-downloading.
    :param filter_string:
        A string to use for filtering runs.

    :return:
        A dataframe of parameters, one row per run. Contains at least the column "run_id".
    """
    # Load parameters
    params_path = buffer_root / "params.tsv"
    if params_path.is_file() and not force:
        logger.info(f"Loading parameters from file {params_path}")
        return pandas.read_csv(params_path, sep="\t")

    logger.info(f"Loading parameters from MLFlow {exp_id}")
    params = get_params_from_experiments(
        experiment_ids=exp_id,
        filter_string=filter_string,
        selection=_has_non_empty_metrics,
        tracking_uri=tracking_uri,
    ).reset_index()
    params.to_csv(params_path, sep="\t", index=False)
    logger.info(f"Saved parameters to {params_path}")
    return params


def get_metric_history_for_runs(
    tracking_uri: str,
    metrics: Union[str, Collection[str]],
    runs: Union[str, Collection[str]],
) -> pandas.DataFrame:
    """
    Get metric history for selected runs.

    :param tracking_uri:
        The URI of the tracking server.
    :param metrics:
        The metrics.
    :param runs:
        The IDs of selected runs.

    :return:
         A dataframe with columns {'run_id', 'key', 'step', 'timestamp', 'value'}.
    """
    # normalize input
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(runs, str):
        runs = [runs]
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    data = []
    task_list = sorted(itertools.product(metrics, runs))
    n_success = n_error = 0
    with tqdm(task_list, unit='metric+task', unit_scale=True) as progress:
        for metric, run in progress:
            try:
                data.extend(
                    (run, measurement.key, measurement.step, measurement.timestamp, measurement.value)
                    for measurement in client.get_metric_history(run_id=run, key=metric)
                )
                n_success += 1
            except ConnectionError as error:
                n_error += 1
                progress.write(f'[Error] {error.strerror}')
            progress.set_postfix(dict(success=n_success, error=n_error))
    return pandas.DataFrame(
        data=data,
        columns=['run_id', 'key', 'step', 'timestamp', 'value']
    )


def buffered_load_metric_history(
    tracking_uri: str,
    metric_names: Collection[str],
    runs: Collection[str],
    buffer_root: pathlib.Path,
    force: bool = False,
) -> pandas.DataFrame:
    """
    Load metric histories from MLflow with TSV-file buffering.

    The buffering is done on a per-metric basis to re-use old buffers when requesting additional metrics.

    :param tracking_uri:
        The tracking URI.
    :param metric_names:
        The names of the metrics.
    :param runs:
        The run IDs.
    :param buffer_root:
        The root directory for buffering. Must be an existing directory.
    :param force:
        Whether to enforce re-downloading.

    :return:
        A dataframe of metric histories in long format, columns: {'run_id', 'key', 'step', 'timestamp', 'value'}.
    """
    metrics = []
    for metric_name in metric_names:
        metrics_path = buffer_root / f"metrics.{metric_name}.tsv"
        if metrics_path.is_file() and not force:
            logger.info(f"Loading metric history from file {metrics_path}")
            single_metric = pandas.read_csv(metrics_path, sep="\t")
        else:
            logger.info(f"Loading metric history for '{metric_name}' from MLFlow")
            single_metric = get_metric_history_for_runs(tracking_uri=tracking_uri, metrics=metric_name, runs=runs)
            single_metric.to_csv(metrics_path, sep="\t", index=False)
            logger.info(f"Saved metric history to {metrics_path}")
        metrics.append(single_metric)
    return pandas.concat(metrics)


def _resolve_experiment_buffer(
    buffer_root: Union[str, pathlib.Path],
    exp_id: int,
    experiment_name: str,
    filter_string: str,
) -> pathlib.Path:
    """Resolve the buffer root for a given experiment and filter string."""
    filter_hash = hashlib.sha512(filter_string.encode(encoding="utf8")).hexdigest()[:8]
    buffer_root = pathlib.Path(buffer_root) / f"{exp_id}_{experiment_name}" / filter_hash
    return buffer_root


def get_results(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "active_learning",
    metric_names: Sequence[str] = ("evaluation.test.hits_at_1", "queries.num_queries"),
    buffer_root: Union[None, str, pathlib.Path] = None,
    force: bool = False,
    filter_string: str = "",
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Get results from MLFlow with local buffering.

    :param tracking_uri:
        The tracking URI.
    :param experiment_name:
        The experiment name.
    :param metric_names:
        The names of the metrics.
    :param buffer_root:
        The root directory for buffering.
    :param force:
        Whether to enforce re-download even if buffered result files exist.
    :param filter_string:
        A filter string to filter for certain runs.

    :return:
        A pair of dataframes with parameters per run, and metrics per run and step.
    """
    mlflow.set_tracking_uri(uri=tracking_uri)
    exp_id = mlflow.get_experiment_by_name(name=experiment_name)
    if exp_id is None:
        raise ValueError(f"{experiment_name} does not exist at MLFlow instance at {tracking_uri}")
    exp_id = exp_id.experiment_id
    logger.info(f"Resolved experiment \"{experiment_name}\": {tracking_uri}/#/experiments/{exp_id}")

    # normalize root
    if buffer_root is None:
        buffer_root = pathlib.Path("/tmp") / "mlflow_buffer"
    buffer_root = _resolve_experiment_buffer(buffer_root, exp_id, experiment_name, filter_string)
    buffer_root.mkdir(exist_ok=True, parents=True)

    params = buffered_load_parameters(
        tracking_uri=tracking_uri,
        exp_id=exp_id,
        buffer_root=buffer_root,
        force=force,
        filter_string=filter_string,
    )
    runs = params["run_id"].tolist()
    logger.info(f'Found {len(runs)} runs.')
    metrics = buffered_load_metric_history(
        tracking_uri=tracking_uri,
        metric_names=metric_names,
        runs=runs,
        buffer_root=buffer_root,
        force=force,
    )
    return params, metrics
