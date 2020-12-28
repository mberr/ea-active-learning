# coding=utf-8
import logging
import random
from collections import defaultdict
from typing import Any, Callable, Collection, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Type

import torch

from ..data import KnowledgeGraphAlignmentDataset, MatchSideEnum, get_other_side
from ..models import KGMatchingModel
from ..modules import Similarity
from ..utils.common import get_subclass_by_name

__all__ = [
    'AlignmentOracle',
    'ModelBasedHeuristic',
    'NodeActiveLearningHeuristic',
    'RandomHeuristic',
    'get_node_active_learning_heuristic_by_name',
    'get_node_active_learning_heuristic_class_by_name',
]


class AlignmentOracle:
    #: The bi-directional training mapping
    _mappings: Mapping[MatchSideEnum, Dict[int, Tuple[int, ...]]]

    #: The available nodes per side.
    _available: Mapping[MatchSideEnum, Set[int]]

    #: The aligned nodes.
    _positives: Mapping[MatchSideEnum, Mapping[int, Set[int]]]

    #: The exclusive nodes per graph
    _negatives: Mapping[MatchSideEnum, Tuple[int, ...]]

    def __init__(self, dataset: KnowledgeGraphAlignmentDataset):
        # build data structure for fast mapping.
        alignment = dataset.alignment
        sides = (MatchSideEnum.left, MatchSideEnum.right)
        mappings = {
            side: defaultdict(list)
            for side in sides
        }
        train_alignment_list = alignment.train.t().tolist()
        for left, right in train_alignment_list:
            mappings[MatchSideEnum.left][left].append(right)
            mappings[MatchSideEnum.right][right].append(left)

        # store immutable
        self._mappings = {
            side: {
                node_id: tuple(matches)
                for node_id, matches in mapping.items()
            }
            for side, mapping in mappings.items()
        }

        # pool consists of nodes part of a training alignment edge OR exclusive
        self._available = {
            side:
            # nodes part of the training alignment
                set(dataset.alignment.train[i].tolist()).union(
                    # exclusive nodes
                    set(range(dataset.num_nodes[side])).difference(dataset.alignment.all[i].tolist())
                )
            for i, side in enumerate(sides)
        }

        # Initially there is no positive or negative information available.
        self._positives = {side: defaultdict(set) for side in sides}
        self._negatives = {side: tuple() for side in sides}

    def label_node(self, node_id: int, side: MatchSideEnum) -> None:
        """Label a node.

        For the node corresponding to node_id on the graph on the given side, the oracle looks up all linked nodes from
        the other side, and either adds the queried node to the set of exclusive nodes of this side (if there are no
        matches), or adds all links from this node to nodes from the other side to the alignment. Afterwards, the node,
        as well as all of its matches are removed from the set of available nodes.

        :param node_id:
            The node ID.
        :param side:
            The side.
        """
        # use batch mode
        self.label_nodes(nodes=[(side, node_id)])

    def label_nodes(self, nodes: Collection[Tuple[MatchSideEnum, int]]) -> None:
        # check availability
        if not all(node_id in self._available[node_side] for (node_side, node_id) in nodes):
            raise AssertionError(f'Requesting unavailable nodes.')

        # Process labelling requests per side for more efficient updates.
        for side in self._mappings.keys():
            # collect labeling requests for this side
            node_ids_this_side = [node_id for (node_side, node_id) in nodes if node_side == side]

            # remove from available
            self._available[side].difference_update(node_ids_this_side)

            for node_id in node_ids_this_side:
                # find match
                matches_other_side = self._mappings[side].get(node_id)

                if matches_other_side is None:
                    self._negatives[side] += (node_id,)
                else:
                    # determine other side
                    other_side = get_other_side(side)

                    # add source -> targets link
                    self._positives[side][node_id].update(matches_other_side)

                    # add targets -> source link
                    for other_node_id in matches_other_side:
                        self._positives[other_side][other_node_id].add(node_id)

                    # remove other nodes from available set
                    self._available[other_side].difference_update(matches_other_side)

    def check_availability(self, side: MatchSideEnum, node_id: int) -> bool:
        return node_id in self._available[side]

    @property
    def num_available(self) -> int:
        """The number of available candidates for labeling."""
        return sum(map(len, self._available.values()))

    @property
    def num_aligned(self) -> Mapping[MatchSideEnum, int]:
        """The number of aligned nodes for each side."""
        return {side: len(mapping.keys()) for side, mapping in self._positives.items()}

    @property
    def num_alignment_pairs(self) -> int:
        """The number of aligned pairs."""
        return sum(len(right_nodes) for left_node, right_nodes in self._positives[MatchSideEnum.left].items())

    @property
    def num_exclusives(self) -> Mapping[MatchSideEnum, int]:
        """The number of known exclusive nodes."""
        return {side: len(nodes) for side, nodes in self._negatives.items()}

    @property
    def available(self) -> Mapping[MatchSideEnum, FrozenSet[int]]:
        """The available nodes per side."""
        # return immutable view
        return {
            side: frozenset(nodes) for side, nodes in self._available.items()
        }

    def restricted_available(
        self,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> Mapping[MatchSideEnum, FrozenSet[int]]:
        """Wrapper around available property to allow restrictions."""
        available = self.available
        if restrict_to is not None:
            available = {
                side: indices.intersection(restrict_to[side])
                for side, indices in available.items()
            }
        return available

    @property
    def available_pairs(self) -> Collection[Tuple[MatchSideEnum, int]]:
        """The available (side, node_id) pairs."""
        return frozenset(sum(([(side, node) for node in nodes] for side, nodes in self._available.items()), []))

    @property
    def positives(self) -> Mapping[MatchSideEnum, Tuple[int, ...]]:
        """The aligned nodes per side."""
        # return immutable view
        return {
            side: tuple(sorted(mapping)) for side, mapping in self._positives.items()
        }

    @property
    def negatives(self) -> Mapping[MatchSideEnum, Tuple[int, ...]]:
        """The exclusive nodes per side."""
        return {
            side: nodes for side, nodes in self._negatives.items()
        }

    @property
    def alignment(self) -> torch.LongTensor:
        """The aligned pairs as tensor.

        :return alignment: shape: (2, num_positives)
        """
        alignment = []
        # as positives stores the alignment for both sides, if suffices to only consider the stored left-to-right mapping
        for left_node_id, right_node_ids in self._positives[MatchSideEnum.left].items():
            alignment += [(left_node_id, right_node_id) for right_node_id in sorted(right_node_ids)]
        return torch.tensor(data=alignment, dtype=torch.long).t()

    @property
    def exclusives(self) -> Mapping[MatchSideEnum, torch.LongTensor]:
        return {
            side: torch.as_tensor(exclusives_on_side, dtype=torch.long)
            for side, exclusives_on_side in self._negatives.items()
        }

    def __str__(self):
        return f'{self.__class__.__name__}(available={self.num_available}, aligned={self.num_aligned}, exclusives={self.num_exclusives}, alignment_pairs={self.num_alignment_pairs})'


class NodeActiveLearningHeuristic:
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning(f'Ignoring positional arguments: {args}')
        if len(kwargs) > 0:
            logging.warning(f'Ignoring keyword-based arguments: {kwargs.keys()}')

    def propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:
        """Batch mode proposal

        :param oracle: The oracle.
        :param num: The number of required nodes.
        :param restrict_to:
            An optional restriction of candidates to sample from.

        :return: A list of tuples (graph, node_id) where
            graph is an enum for the side
            node_id corresponds to the selected node in the graph.
        """
        raise NotImplementedError


class RandomHeuristic(NodeActiveLearningHeuristic):
    """Randomly proposes the next node to label."""

    def propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        return self.random_selection(oracle=oracle, num=num, restrict_to=restrict_to)

    @staticmethod
    def random_selection(
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:
        return random.sample(list(
            (side, node_id)
            for side, available_on_side in oracle.restricted_available(restrict_to=restrict_to).items()
            for node_id in available_on_side
        ), num)


class ModelBasedHeuristic(NodeActiveLearningHeuristic):
    """A node active learning heuristic using the model output for decision."""

    def __init__(self, model: KGMatchingModel, similarity: Similarity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.similarity = similarity
        self.fallback_heuristic = RandomHeuristic()

    def propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:
        try:
            # Set model into evaluation mode
            self.model.eval()

            with torch.no_grad():
                return self._propose_next_nodes(oracle=oracle, num=num, restrict_to=restrict_to)[:num]
        except KeyError:
            # For the first iteration the model has not been trained and hence is not able to do a prediction
            return self.fallback_heuristic.propose_next_nodes(oracle=oracle, num=num, restrict_to=restrict_to)

    def _propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:
        raise NotImplementedError


def _default_heuristic_name_normalizer(name: str) -> str:
    return name.replace('Heuristic', '').replace('Centrality', '').lower()


def get_node_active_learning_heuristic_class_by_name(
    name: str,
    normalizer: Optional[Callable[[str], str]] = None,
) -> Type[NodeActiveLearningHeuristic]:
    if normalizer is None:
        normalizer = _default_heuristic_name_normalizer
    return get_subclass_by_name(base_class=NodeActiveLearningHeuristic, name=name, normalizer=normalizer)


def get_node_active_learning_heuristic_by_name(
    name: str,
    **kwargs: Any,
) -> NodeActiveLearningHeuristic:
    """
    Factory method for node active learning heuristics.

    :param name:
        The name of the heuristic.
    :param kwargs:
        The keyword-based arguments.

    :return:
        An instance of a node active learning heuristic.
    """
    # Get heuristic class
    cls = get_node_active_learning_heuristic_class_by_name(name=name)

    # instantiate
    return cls(**kwargs)
