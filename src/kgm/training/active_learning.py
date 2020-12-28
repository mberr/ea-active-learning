import logging
from typing import Iterator, Mapping, Any, Tuple

import torch
import tqdm

from .matching import AlignmentModelTrainer
from ..active_learning import AlignmentOracle, NodeActiveLearningHeuristic
from ..data import KnowledgeGraphAlignmentDataset
from ..eval import evaluate_model
from ..models import KGMatchingModel
from ..modules import Similarity


def evaluate_active_learning_heuristic(
    dataset: KnowledgeGraphAlignmentDataset,
    model: KGMatchingModel,
    similarity: Similarity,
    heuristic: NodeActiveLearningHeuristic,
    trainer: AlignmentModelTrainer,
    step_size: int,
    eval_batch_size: int,
    remove_exclusives: bool = True,
    restart: bool = False,
) -> Iterator[Tuple[int, Mapping[str, Any]]]:
    """
    Evaluate an active learning heuristic.

    :param dataset:
        The dataset.
    :param model:
        The model.
    :param similarity:
        The similarity.
    :param heuristic:
        The heuristic.
    :param trainer:
        The trainer.
    :param step_size:
        The step-size for batch-mode query generation.
    :param eval_batch_size:
        The evaluation batch size.
    :param remove_exclusives:
        Whether to remove exclusive nodes from the graph.
    :param restart:
        Whether to re-initialize the model after each query generation.

    :return:
        yields (step, evaluation) pairs, where evaluation is a dictionary containing the evaluation results.
    """
    # create alignment oracle
    oracle = AlignmentOracle(dataset=dataset)
    logging.info(f'Created oracle: {oracle}')
    max_queries = oracle.num_available

    # Evaluate
    logging.info('Begin to evaluate model-based active learning heuristic.')
    query_counter = 0
    total_epoch = 0
    for step in tqdm.trange(step_size, max_queries + 1, step_size, desc=f'Evaluating {heuristic.__class__.__name__} on {dataset.dataset_name}:{dataset.subset_name}'):
        remaining = oracle.num_available
        if remaining <= 0:
            logging.info('No more candidates available.')
            break

        # query
        with torch.no_grad():
            # determine number of queries
            num = min(step_size, remaining)

            # add to query counter
            query_counter += num

            # propose queries
            queries = heuristic.propose_next_nodes(oracle=oracle, num=num)
            if len(queries) < num:
                raise AssertionError(queries)

            # batch-mode labeling
            oracle.label_nodes(nodes=queries)

            # prepare for training
            train_alignment = oracle.alignment
            exclusives = oracle.exclusives if remove_exclusives else None

        if restart:
            model.reset_parameters()

        if train_alignment.numel() == 0:
            logging.fatal(f'After {step} queries, not a single alignment is available. That should not happen often.')
            continue

        result = trainer.train(
            edge_tensors=dataset.edge_tensors,
            train_alignment=train_alignment,
            exclusives=exclusives,
            validation_alignment=dataset.alignment.validation,
            keep_all=False,
        )[0]
        total_epoch += result['epoch']
        result['total_epoch'] = total_epoch
        result['evaluation'] = evaluate_model(
            model=model,
            alignments=dataset.alignment.to_dict(),
            similarity=similarity,
            eval_batch_size=eval_batch_size,
        )
        result['queries'] = dict(
            num_queries=query_counter,
            num_alignment_pairs=train_alignment.shape[1],
            num_exclusives=sum(e.shape[0] for e in exclusives.values()) if exclusives is not None else sum(oracle.num_exclusives.values()),
        )
        yield step, result

    if dataset.alignment.train.shape != oracle.alignment.shape:
        logging.warning(f'Shape mismatch for final alignment: dataset.alignment.train.shape={dataset.alignment.train.shape} vs. oracle.alignment.shape={oracle.alignment.shape}.')
    if oracle.num_exclusives != dataset.num_exclusives:
        logging.warning(f'Number of exclusives differs for oracle and dataset: oracle.num_exclusives={oracle.num_exclusives} vs. dataset.num_exclusives={dataset.num_exclusives}')
