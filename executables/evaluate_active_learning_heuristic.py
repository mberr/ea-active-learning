import argparse
import copy
import logging
import mlflow
import torch
from torch import optim
from typing import Any, List, Mapping, Optional, Tuple

from kgm.active_learning import get_node_active_learning_heuristic_by_name
from kgm.data import KnowledgeGraphAlignmentDataset, get_dataset_by_name
from kgm.data.edge_modifiers import EdgeModifierHeuristic, RemoveEdgesHeuristic
from kgm.data.knowledge_graph import available_datasets
from kgm.models import EdgeWeightsEnum, GCNAlign, KGMatchingModel
from kgm.modules import MarginLoss, MatchingLoss, SampledMatchingLoss, Similarity, get_similarity
from kgm.modules.embeddings.init.base import NodeEmbeddingInitMethod
from kgm.training.active_learning import evaluate_active_learning_heuristic
from kgm.training.matching import AlignmentModelTrainer, EarlyStoppingTrainer
from kgm.utils.common import argparse_bool, from_dot, generate_experiments, to_dot
from kgm.utils.mlflow_utils import log_metrics_to_mlflow, run_experiments


def evaluate_heuristic(
    eval_batch_size: int,
    heuristic_name: str,
    model: KGMatchingModel,
    remove_exclusives: bool,
    similarity: Similarity,
    step_size: int,
    trainer: AlignmentModelTrainer,
    dataset: KnowledgeGraphAlignmentDataset,
    restart: bool = False,
    heuristic_kwargs: Optional[Mapping[str, Any]] = None,
):
    # initialize heuristic
    if heuristic_kwargs is None:
        heuristic_kwargs = {}
    heuristic = get_node_active_learning_heuristic_by_name(
        name=heuristic_name,
        dataset=dataset,
        model=model,
        similarity=similarity,
        dataset_name=dataset.dataset_name,
        subset_name=dataset.subset_name,
        **heuristic_kwargs
    )

    for step, result in evaluate_active_learning_heuristic(
        dataset=dataset,
        model=model,
        similarity=similarity,
        heuristic=heuristic,
        trainer=trainer,
        step_size=step_size,
        eval_batch_size=eval_batch_size,
        remove_exclusives=remove_exclusives,
        restart=restart,
    ):
        log_metrics_to_mlflow(metrics=result, step=step)


def active_learning_heuristic_experiment(
    params: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], int]:
    active_learning_evaluation, dataset, dataset_name, edge_modifiers, loss, model, params, similarity, subset_name = _common_part(params)

    # Trainer
    early_stopper_params = params['early_stopping']
    eval_batch_size = early_stopper_params['eval_batch_size']
    trainer = EarlyStoppingTrainer(
        model=model,
        similarity=similarity,
        loss=loss,
        edge_modifiers=edge_modifiers,
        **params['early_stopping']
    )

    # Evaluate heuristic
    heuristic_params = params['heuristic']
    heuristic_name = heuristic_params['heuristic_name']
    heuristic_kwargs = heuristic_params.get('heuristic_kwargs', {})
    evaluate_heuristic(
        dataset=dataset,
        similarity=similarity,
        model=model,
        trainer=trainer,
        eval_batch_size=eval_batch_size,
        heuristic_name=heuristic_name,
        heuristic_kwargs=heuristic_kwargs,
        **active_learning_evaluation,
    )

    return {}, 0


def _common_part(params) -> Tuple[
    Mapping,
    KnowledgeGraphAlignmentDataset,
    str,
    List[EdgeModifierHeuristic],
    MatchingLoss,
    GCNAlign,
    Mapping,
    Similarity,
    str,
]:
    # convert to nested dictionary
    params = from_dot(params)
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        logging.warning('Falling back to CPU.')
        device = 'cpu'
    device = torch.device(device=device)
    # Load dataset
    dataset_params = params['data']
    dataset_name = dataset_params['dataset_name']
    subset_name = dataset_params['subset_name']
    dataset = get_dataset_by_name(**dataset_params)
    # create model
    params['model'] = params['model']
    model = GCNAlign(
        num_nodes=dataset.num_nodes,
        device=device,
        **params['model']
    )
    # construct similarity
    similarity_params = params['similarity']
    similarity = get_similarity(**similarity_params)
    # construct loss
    loss_params = params['loss']
    loss = SampledMatchingLoss(
        similarity=similarity,
        pairwise_loss=MarginLoss(margin=loss_params['margin']),
        num_negatives=loss_params['num_negatives'],
    )
    # send to device
    model = model.to(device=device)
    loss = loss.to(device=device)
    active_learning_evaluation = params['active_learning_evaluation']
    if active_learning_evaluation['remove_exclusives']:
        edge_modifiers = [RemoveEdgesHeuristic()]
    else:
        edge_modifiers = []
    return active_learning_evaluation, dataset, dataset_name, edge_modifiers, loss, model, params, similarity, subset_name


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    all_datasets = available_datasets()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wk3l15k', choices=list(all_datasets.keys()))
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--train_validation_ratio', type=float, default=0.8)
    parser.add_argument('--restart', type=argparse_bool, default=False)
    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000')
    parser.add_argument('--phase', type=str, choices=["random", "hpo", "best"], default="best")
    args = parser.parse_args()

    eval_batch_size = 2048

    # Mlflow settings
    logging.info(f'Logging to MLFlow @ {args.tracking_uri}')
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(experiment_name='active_learning')

    dataset_name = args.dataset
    subset_name = args.subset
    if subset_name is None:
        subset_name = sorted(all_datasets[dataset_name])

    base_config = dict(
        data=dict(
            dataset_name=dataset_name,
            subset_name=subset_name,
            inverse_triples=True,  # GCNAlign default
            self_loops=True,  # GCNAlign default
            train_validation_split=args.train_validation_ratio,
        ),
        model=dict(
            embedding_dim=200,
            n_layers=2,
            use_conv_weights=False,
            edge_weight_mode=EdgeWeightsEnum.inverse_in_degree,
            node_embedding_init=NodeEmbeddingInitMethod.std_one,
            node_embedding_dropout=None,  # [None, 0.2, 0.5],
        ),
        similarity=dict(
            similarity='l1',
            transformation='negative',
        ),
        loss=dict(
            margin=3.,
            num_negatives=50,
        ),
        early_stopping=dict(
            num_epochs=4_000,
            eval_frequency=20,
            eval_batch_size=eval_batch_size,
            optimizer_cls=optim.Adam,
            optimizer_kwargs={'lr': 1.0, },
            patience=10,
            eval_key='mrr',
            larger_is_better=True,
        ),
        active_learning_evaluation=dict(
            step_size=200,
            remove_exclusives=True,
            # remove_exclusives=[False, True],
            restart=False,
        ),
    )

    # 1. Random heuristic: Robust estimates
    if args.phase == "random":
        num_replicates = 5
        heuristic_grids = {
            'random': {}
        }

    # 2. Initial study
    elif args.phase == "hpo":
        # The grid for each heuristic
        csls_k = [None, 2]
        use_max = [False, True]
        softmax_temperature = [0.01, 0.1, None, 10., 100., ]
        prob_diff_threshold = [0.05, 0.1, 0.2, 0.5, 0.7]
        steps_to_look_back = [1, 2, 3, 5, 10]

        heuristic_grids = {
            'random': {},
            # embedding space cover
            'coreset': {},
            # graph cover
            'approximatevertexcover': {},
            # centrality
            'betweenness': {},
            'degree': {},
            # learning
            'PreviousExperienceBased': {
                'csls_k': csls_k,
                'prob_diff_threshold': prob_diff_threshold,
                'steps_to_look_back': steps_to_look_back,
            },
            # bayesian
            'BALD': {
                'temperature': softmax_temperature,
                'use_max_entropy': use_max,
            },
            'MostProbableMatchingInUnexploredRegion': dict(
                number_centroids=[4, 16, 32, 64, 128, 256, 512],
                consider_negatives_as_labeled=[False, True],
                normalize_by_cluster_size=[False, True],
                matching_heuristic=['random', 'degree', 'maxsimilarity', 'learning'],
            )
        }
        num_replicates = 1

    elif args.phase == "best":

        # Best HPO config with multiple replicates
        num_replicates = 5
        heuristic_grids = {
            'BALD': dict(
                dropout=0.2,
                use_max_entropy=True,
                temperature=0.01,
            ),
            'MostProbableMatchingInUnexploredRegion': dict(
                dropout=None,
                consider_negatives_as_labeled=True,
                matching_heuristic='degree',
                normalize_by_cluster_size=True,
                number_centroids=256,
            ),
            'PreviousExperienceBased': dict(
                dropout=None,
                csls_k=2,
                prob_diff_threshold=0.2,
                steps_to_look_back=1,
            ),
            'approximatevertexcover': dict(
                dropout=None,
            ),
            'betweenness': dict(
                dropout=None,
            ),
            'coreset': dict(
                dropout=None,
            ),
            'degree': dict(
                dropout=None,
            ),
            # 'random': dict(),  # Already done before
        }
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

    # Generate search runs from per-heuristic grid
    experiments = []
    for name, kwargs in heuristic_grids.items():
        this_grid = copy.deepcopy(base_config)
        dropout = kwargs.pop('dropout', None)
        if dropout is not None:
            this_grid['model']['node_embedding_dropout'] = dropout
        this_grid['heuristic'] = dict(
            heuristic_name=name,
            heuristic_kwargs=kwargs,
        )
        experiments += generate_experiments(
            grid_params=to_dot(this_grid, function_to_name=False),
            explicit=None,
        )

    run_experiments(
        search_list=experiments,
        experiment=active_learning_heuristic_experiment,
        num_replicates=num_replicates,
        break_on_error=True,
    )


if __name__ == '__main__':
    main()
