# coding=utf-8
import hashlib
import logging
import os
import platform
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import mlflow
import mlflow.entities

from kgm.utils.common import to_dot


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
