# coding=utf-8
import logging
from typing import Collection, List, Mapping, MutableMapping, Optional, Tuple, Type, Union

import torch
from torch.nn import functional

from . import AlignmentOracle, ModelBasedHeuristic, NodeActiveLearningHeuristic, RandomHeuristic, get_node_active_learning_heuristic_by_name
from ..data import MatchSideEnum, SIDES, get_other_side
from ..models import KGMatchingModel
from ..modules import Similarity
from ..modules.similarity import generalized_k_means
from ..utils.torch_utils import csls, softmax_entropy_from_logits

__all__ = [
    'BatchOptimizedMaxSimilarityHeuristic',
    'CoreSetHeuristic',
    'MaxSimilarityHeuristic',
    'MinMaxSimilarityHeuristic',
    'OneVsAllBinaryEntropyHeuristic',
    'SimilarityBasedHeuristic',
]


class BaseMinMaxSimilarityHeuristic(ModelBasedHeuristic):
    """Base class for core-set and MinMaxSimilarity heuristic."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        on_same_side: bool,
        **kwargs,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            **kwargs
        )
        self.same_side = on_same_side

    def _propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        # get representations
        node_repr = self.model.forward()

        # for each unlabeled points determine the most similar labeled point on correct side
        unlabeled_to_labeled: MutableMapping[MatchSideEnum, MutableMapping[int, float]] = {}
        for side, unlabeled_this_side in oracle.restricted_available(restrict_to=restrict_to).items():
            unlabeled_this_side = sorted(unlabeled_this_side)
            labeled_side = side if self.same_side else get_other_side(side=side)
            labeled = sorted(oracle.positives[labeled_side])
            if len(labeled) == 0:
                unlabeled_match_sim = torch.empty(len(unlabeled_this_side)).fill_(value=float('inf'))
            else:
                unlabeled_match_sim = self.similarity.all_to_all(
                    left=node_repr[side][unlabeled_this_side],
                    right=node_repr[labeled_side][labeled],
                ).max(dim=1)[0]
            assert len(unlabeled_this_side) == unlabeled_match_sim.shape[0]
            unlabeled_to_labeled[side] = {
                unlabeled_id: labeled_match_sim
                for unlabeled_id, labeled_match_sim in zip(unlabeled_this_side, unlabeled_match_sim.tolist())
            }

        selection = []
        for _ in range(num):
            # determine unlabeled point with least similarity to its most similar labeled point
            selected = side1, id1 = min(
                (sim, side1, unlabeled1)
                for side1, matches in unlabeled_to_labeled.items()
                for unlabeled1, sim in matches.items()
            )[1:]

            if side1 is None:
                logging.error(f'No more candidates available. selection={selection}, oracle.available={oracle.available}')
                raise RuntimeError

            # add unlabeled point to labeled, and make it unavailable
            selection.append(selected)
            unlabeled_to_labeled[side1].pop(id1)

            # compute similarity from newly labeled to all remaining unlabeled
            this_repr = node_repr[side1][id1].unsqueeze(0)
            unlabeled_available = sorted(unlabeled_to_labeled[side1].keys())
            if len(unlabeled_available) == 0:
                # if no more nodes are available, there remains nothing to be updated
                continue

            unlabeled_repr = node_repr[side1][unlabeled_available]
            this_sim_to_other_unlabeled = self.similarity.all_to_all(
                left=unlabeled_repr,
                right=this_repr
            )[:, 0]

            # update matching similarities
            for other_id, new_sim in zip(unlabeled_available, this_sim_to_other_unlabeled.tolist()):
                old_sim = unlabeled_to_labeled[side1][other_id]
                if old_sim < new_sim:
                    unlabeled_to_labeled[side1][other_id] = new_sim
        return selection


class MinMaxSimilarityHeuristic(BaseMinMaxSimilarityHeuristic):
    """Selects nodes with the smallest similarity to the most similar matching nodes."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            on_same_side=False,
            *args,
            **kwargs
        )


class CoreSetHeuristic(BaseMinMaxSimilarityHeuristic):
    """Selects nodes according to https://arxiv.org/pdf/1708.00489.pdf"""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            on_same_side=True,
            *args,
            **kwargs
        )


class SimilarityBasedHeuristic(ModelBasedHeuristic):
    """A node active learning heuristic using the model output for decision."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        csls_k: Optional[int] = 1,
        use_max_sim: bool = True,
        **kwargs,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            **kwargs
        )
        self.csls_k = csls_k
        self.use_max_sim = use_max_sim

    def _propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        # get representations
        node_repr = self.model.forward()

        selection = []
        for side, indices in oracle.restricted_available(restrict_to=restrict_to).items():
            indices_this_side = list(indices)
            this_repr = node_repr[side][indices_this_side]
            other_side = get_other_side(side=side)
            all_other_repr = node_repr[other_side]
            this_to_all_other_sim = self.similarity.all_to_all(left=this_repr, right=all_other_repr)
            if self.csls_k is not None:
                this_to_all_other_sim = csls(sim=this_to_all_other_sim, k=self.csls_k)
            selection += [
                (value, side, indices_this_side[node_id_id])
                for (value, node_id_id) in self.select_from_similarity(sim=this_to_all_other_sim)
            ]
        return [s[1:] for s in sorted(selection, reverse=self.use_max_sim)][:num]

    def select_from_similarity(
        self,
        sim: torch.FloatTensor,
    ) -> List[Tuple[float, int]]:
        """Selects nodes given the similarity of these to all other from the other side.

        :param sim: shape: (num_available_on_side, num_all_other_side)
            The similarity matrix.
        :return:
        """
        score = self.score_nodes(sim=sim)
        values, node_indices = score.sort(dim=0, descending=True)
        return list(zip(values.cpu().tolist(), node_indices.cpu().tolist()))

    def score_nodes(
        self,
        sim: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute a score for each node.

        :param sim: shape: (num_available_on_side, num_all_other_side)
            The similarity matrix.
        :param oracle:
            The oracle allowing access to exclusive and aligned nodes.

        :return: shape: (num_available_on_side,)
            A score for each node. Larger is better.
        """
        raise NotImplementedError


class MaxSimilarityHeuristic(SimilarityBasedHeuristic):
    """Selects nodes with the maximum similarity."""

    def score_nodes(
        self,
        sim: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return sim.max(dim=1)[0]


class SoftmaxEntropyHeuristic(SimilarityBasedHeuristic):
    """Selects nodes according to high entropy of the softmax distribution over similarities."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        use_max_sim: bool = True,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model=model, similarity=similarity, use_max_sim=use_max_sim, **kwargs)
        self.temperature = temperature
        self.k = top_k

    def score_nodes(
        self,
        sim: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        if self.k is not None:
            sim = sim.topk(k=self.k, dim=1, sorted=False)[0]
        return softmax_entropy_from_logits(sim, dim=1, temperature=self.temperature)


class OneVsAllBinaryEntropyHeuristic(SimilarityBasedHeuristic):
    """Selects nodes according to the entropy of the highest-similarity-vs-all sigmoid distribution."""

    def score_nodes(
        self,
        sim: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        max_sim = sim.max(dim=1)[0]
        return functional.binary_cross_entropy_with_logits(input=max_sim, target=torch.ones_like(max_sim), reduction='none')


class BatchOptimizedMaxSimilarityHeuristic(ModelBasedHeuristic):
    """Selects nodes with the maximum similarity. For likely matches only proposes one of them."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        csls_k: int = 0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            **kwargs
        )
        self.csls_k = csls_k

    def _propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        # get representations
        node_repr = self.model.forward()

        selection: List[Tuple[float, Tuple[MatchSideEnum, int], Tuple[MatchSideEnum, int]]] = []
        for side, indices in oracle.restricted_available(restrict_to=restrict_to).items():
            if len(indices) == 0:
                # no more nodes available
                continue
            # compute similarity
            indices_this_side = list(indices)
            this_repr = node_repr[side][indices_this_side]
            other_side = get_other_side(side=side)
            all_other_repr = node_repr[other_side]
            this_to_all_other_sim = self.similarity.all_to_all(left=this_repr, right=all_other_repr)

            this_to_all_other_sim = csls(sim=this_to_all_other_sim, k=self.csls_k)

            # greedy matching
            match_sim, match_ind = this_to_all_other_sim.max(dim=1)
            selection.extend(
                (sim, (side, ind), (other_side, other_ind))
                for sim, ind, other_ind in zip(match_sim.tolist(), indices_this_side, match_ind.tolist())
            )

        # sort by similarity
        result = []
        skip = set()
        for (_, choice, match) in sorted(selection, reverse=True):
            # skip those which likely match already chosen nodes
            if choice in skip:
                continue

            # add choice to result
            result.append(choice)

            # add likely match to skip list
            skip.add(match)

            # terminate when there are enough chosen nodes
            if len(result) >= num:
                break

        # if there are not enough candidates to propose
        missing = num - len(result)
        if missing > 0:
            result += list(skip)[:missing]

        return result


def _cluster_node_representations_in_joint_space(
    node_repr: Mapping[MatchSideEnum, torch.FloatTensor],
    similarity: Similarity,
    number_centroids: int,
    num_iterations: int,
    tolerance: float,
) -> Mapping[MatchSideEnum, torch.BoolTensor]:
    """
    Cluster node representations in joint space.

    :param node_repr:
        The node representations.
    :param similarity:
        The similarity.
    :param number_centroids:
        The number of centroids.
    :param num_iterations:
        The maximum number of iterations.
    :param tolerance:
        A tolerance for convergence check.

    :return:
        A mapping side -> cluster where
            cluster: shape: (num_nodes_on_side, num_centroids)
                cluster[i, j] = True iff x_i belongs to C_j
    """
    # since both embeddings  end up in the joined space, cluster them together
    assignment = generalized_k_means(
        x=torch.cat([node_repr[side] for side in SIDES], dim=0),
        similarity=similarity,
        k=number_centroids,
        max_iter=num_iterations,
        tolerance=tolerance,
    )[0]

    return {
        side: partial_assignment[:, None] == torch.arange(number_centroids, dtype=torch.long, device=assignment.device)[None, :]
        for side, partial_assignment in zip(SIDES, assignment.split([node_repr[side].shape[0] for side in SIDES]))
    }


def _create_one_hot_available_per_cluster_matrix(
    clusters: Mapping[MatchSideEnum, torch.BoolTensor],
    oracle: AlignmentOracle,
    restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]],
) -> Mapping[MatchSideEnum, torch.BoolTensor]:
    """
    Create a one-hot matrix indicating the available nodes per cluster.

    :param clusters:
        The clusters as mapping side -> C, where
            C.shape = (num_nodes_on_side, num_clusters), and C[i, j] = 1 iff x_i belongs to cluster C_j.
    :param oracle:
        The oracle.
    :param restrict_to:
        Restrict available nodes to these node IDs.

    :return:
    """
    # Get available nodes per side
    available = oracle.restricted_available(restrict_to=restrict_to)

    # Transform into boolean tensor, shape: (num_nodes_on_side,)
    available = {
        side: clusters[side].new_zeros(
            clusters[side].shape[0],
            dtype=torch.bool,
        ).scatter_(
            dim=0,
            index=torch.as_tensor(list(indices), dtype=torch.long, device=clusters[side].device),
            value=True,
        )
        for side, indices in available.items()
    }

    # Mask for available nodes per cluster, shape: (num_nodes_on_side, num_clusters)
    available_per_cluster = {
        side: clusters[side] & available_on_side[:, None]
        for side, available_on_side in available.items()
    }

    return available_per_cluster


class MostProbableMatchingInUnexploredRegionHeuristic(ModelBasedHeuristic):
    """
    The heuristic clusters embeddings and samples from each cluster according to number or fraction of already labeled objects in the cluster.
    The more labeled objects are already in the cluster, the less is the probability to sample from the cluster again.
    Nodes with the highest probability to have matching are selected from the cluster
    """

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        number_centroids: int = 10,
        consider_negatives_as_labeled: bool = True,
        normalize_by_cluster_size: bool = True,
        num_iterations: int = 2000,
        tolerance: float = 1e-08,
        matching_heuristic: Optional[Union[str, NodeActiveLearningHeuristic, Type[NodeActiveLearningHeuristic]]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            **kwargs
        )
        if matching_heuristic is None:
            matching_heuristic = RandomHeuristic()
        elif isinstance(matching_heuristic, type):
            matching_heuristic = matching_heuristic(
                model=model,
                similarity=similarity,
                **kwargs,
            )
        elif isinstance(matching_heuristic, str):
            matching_heuristic = get_node_active_learning_heuristic_by_name(
                name=matching_heuristic,
                model=model,
                similarity=similarity,
                **kwargs
            )
        else:
            raise ValueError(matching_heuristic)
        self.number_centroids = number_centroids
        self.matching_heuristic = matching_heuristic
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.consider_negatives_as_labeled = consider_negatives_as_labeled
        self.normalize_by_cluster_size = normalize_by_cluster_size

    def _compute_number_of_samples_per_cluster(
        self,
        oracle: AlignmentOracle,
        clusters: Mapping[MatchSideEnum, torch.BoolTensor],
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> Tuple[Optional[List[int]], List[Mapping[MatchSideEnum, Collection[int]]]]:
        """
        Given a clustering, compute the number of samples per cluster.

        This is the heart-piece of this heuristic.

        :param oracle:
            The oracle.
        :param clusters:
            The clusters.
        :param num:
            The total number of samples.
        :param restrict_to:
            Restrict available nodes to these nodes.

        :return:
            The number of samples to draw from each cluster.
        """
        available_per_cluster = _create_one_hot_available_per_cluster_matrix(clusters=clusters, oracle=oracle, restrict_to=restrict_to)
        assert all(clusters[side].shape[0] == num_nodes_on_side for side, num_nodes_on_side in self.model.num_nodes.items())

        # Number of available nodes per cluster, shape: (num_clusters,)
        num_avail_per_cluster: torch.LongTensor = sum(available_in_cluster.sum(dim=0) for available_in_cluster in available_per_cluster.values())
        if num_avail_per_cluster.sum() < num:
            raise AssertionError

        # Check number of labelled nodes per cluster
        cluster_score = num_avail_per_cluster.new_zeros(self.number_centroids)
        for side in SIDES:
            indices = oracle.positives[side]
            if self.consider_negatives_as_labeled:
                indices = indices + oracle.negatives[side]
            indices = torch.as_tensor(indices, dtype=torch.long, device=cluster_score.device)
            cluster_score += clusters[side].index_select(dim=0, index=indices).sum(dim=0)

        # Optionally normalize by cluster size
        cluster_score = cluster_score.float()
        if self.normalize_by_cluster_size:
            cluster_score *= sum(clusters_on_side.sum(dim=0) for side, clusters_on_side in clusters.items()).clamp_min(1).float().reciprocal()

        # normalize weights
        sample_weights = functional.softmax(-cluster_score, dim=0)

        # Rejection sampling
        missing = num
        num_samples_per_cluster = cluster_score.new_zeros(self.number_centroids, dtype=torch.long)
        while missing > 0:
            new_samples = torch.multinomial(sample_weights, num_samples=missing, replacement=True)
            num_samples_per_cluster[new_samples] += 1
            num_samples_per_cluster = torch.min(num_samples_per_cluster, num_avail_per_cluster)
            missing = num - num_samples_per_cluster.sum()

        available_per_cluster = [
            {
                side: indices[:, c_id].nonzero().squeeze(dim=1).tolist()
                for side, indices in available_per_cluster.items()
            }
            for c_id in range(self.number_centroids)
        ]
        return num_samples_per_cluster.tolist(), available_per_cluster

    def _propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:
        # Cluster node representations
        node_repr = self.model.forward(indices=None)
        assert all(node_repr_on_side.shape[0] == self.model.num_nodes[side] for side, node_repr_on_side in node_repr.items())

        clusters = _cluster_node_representations_in_joint_space(
            node_repr=node_repr,
            similarity=self.similarity,
            number_centroids=self.number_centroids,
            num_iterations=self.num_iterations,
            tolerance=self.tolerance,
        )
        assert all(clusters[side].shape[0] == num for side, num in self.model.num_nodes.items())
        # release memory
        del node_repr

        # Compute number of samples per cluster
        num_samples_per_cluster, available_per_cluster = self._compute_number_of_samples_per_cluster(
            oracle=oracle,
            clusters=clusters,
            num=num,
            restrict_to=restrict_to,
        )

        # Sample according to base heuristic
        return sum(
            (
                self.matching_heuristic.propose_next_nodes(oracle=oracle, num=cluster_num, restrict_to=available)
                for available, cluster_num in zip(available_per_cluster, num_samples_per_cluster)
            ),
            []
        )
