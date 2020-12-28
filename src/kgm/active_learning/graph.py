# coding=utf-8
import itertools
import logging
import os
from os import path
from typing import Collection, Generic, List, Mapping, Optional, Tuple, TypeVar

import networkx
import numpy
from scipy import sparse
from scipy.sparse.csgraph import floyd_warshall

from . import AlignmentOracle, NodeActiveLearningHeuristic, RandomHeuristic
from ..data import KnowledgeGraph, KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES

T = TypeVar('T')

__all__ = [
    'ApproximateVertexCoverHeuristic',
    'BetweennessCentralityHeuristic',
    'BufferedHeuristic',
    'CentralityHeuristic',
    'ClosenessCentralityHeuristic',
    'DegreeCentralityHeuristic',
    'HarmonicCentralityHeuristic',
    'MaximumShortestPathDistanceHeuristic',
    'PageRankCentralityHeuristic',
]


class BufferedHeuristic(NodeActiveLearningHeuristic, Generic[T]):
    """A heuristic which buffers some values across different runs."""

    # The buffer
    buffer: T

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        artifact_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if artifact_root is None:
            artifact_root = path.join('..', 'artifacts')
        self.dataset_name = dataset.dataset_name
        self.subset_name = dataset.subset_name
        self.artifact_root = artifact_root
        if dataset.dataset_name is not None:
            artifact_root = path.join(artifact_root, dataset.dataset_name)
        if dataset.subset_name is not None:
            artifact_root = path.join(artifact_root, dataset.subset_name)
        self.buffer = self._load(
            dataset=dataset,
            artifact_root=artifact_root,
        )

    def _load(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        artifact_root: str,
    ) -> T:
        raise NotImplementedError


class ApproximateVertexCoverHeuristic(BufferedHeuristic[List[Tuple[MatchSideEnum, int]]]):
    """
    Propose nodes based on approximate vertex cover following

    Efficient Algorithms for Social Network Coverage and Reach
    Deepak Puthal, Surya Nepal, Cecile Paris, Rajiv Ranjan and Jinjun Chen
    2015 IEEE International Congress on Big Data
    """

    def _load(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        artifact_root: str,
    ) -> List[Tuple[MatchSideEnum, int]]:
        artifact_path = path.join(artifact_root, 'avc.npy')
        if path.isfile(artifact_path):
            ranking = numpy.load(artifact_path)
            logging.info(f'Loaded scores from {artifact_path}.')
        else:
            logging.info(f'{artifact_path} not existing. Thus, computing scores.')
            os.makedirs(artifact_root, exist_ok=True)

            # convert to networkx graphs
            graphs = {
                side: graph.to_networkx()
                for side, graph in dataset.graphs.items()
            }

            # initialize weights by degree
            weights = {
                (side, node): degree
                for side, graph in graphs.items()
                for node, degree in graph.degree()
            }

            ranking = []
            while len(weights) > 0:
                # get node with highest weight
                (side, node), _ = max(weights.items(), key=lambda item: item[1])

                # decrease weight of neighbors
                graph = graphs[side]
                for neighbor in graph.neighbors(node):
                    weights[(side, neighbor)] -= 1

                # delete node
                graph.remove_node(node)
                del weights[(side, node)]

                # append node to ranking
                ranking.append([SIDES.index(side), node])
            numpy.save(artifact_path, ranking)

        return [(SIDES[side_id], node_id) for side_id, node_id in ranking]

    def propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:
        available = oracle.restricted_available(restrict_to=restrict_to)

        def _filter(item: Tuple[MatchSideEnum, int]):
            side, node_id = item
            return node_id in available[side]

        return list(itertools.islice(filter(_filter, self.buffer), num))


class CentralityHeuristic(BufferedHeuristic[Mapping[MatchSideEnum, List[Tuple[float, int]]]]):
    """Propose nodes based on centrality measure."""

    def _load(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        artifact_root: str,
    ) -> Mapping[MatchSideEnum, List[Tuple[float, int]]]:
        result = {}
        class_name = self.__class__.__name__.lower()
        for side, graph in dataset.graphs.items():
            artifact_path = path.join(artifact_root, f'{class_name}_{str(side.value)}.npy')
            if not path.isfile(artifact_path):
                logging.info(f'{artifact_path} not existing. Thus, computing scores.')
                directory = path.dirname(artifact_path)
                os.makedirs(directory, exist_ok=True)
                centrality = self.centrality(graph=graph.to_networkx())
                scores = numpy.asarray(list(centrality[i] for i in range(graph.num_entities)))
                numpy.save(artifact_path, scores)
            else:
                scores = numpy.load(artifact_path)
                logging.info(f'Loaded scores from {artifact_path}.')
            sort_indices = scores.argsort()[::-1]
            result[side] = list(zip(scores[sort_indices].tolist(), sort_indices.tolist()))
        return result

    # pylint: disable=cell-var-from-loop
    def propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        available = oracle.restricted_available(restrict_to=restrict_to)
        candidates = []
        for side, scoring in self.buffer.items():
            available_on_side = available[side]

            def _filter(candidate: Tuple[float, int]) -> bool:
                return candidate[1] in available_on_side

            candidates.extend((score, side, node) for (score, node) in itertools.islice(filter(_filter, scoring), num))

        return list(itertools.islice(((side, node) for (score, side, node) in sorted(candidates, reverse=True)), num))

    def centrality(self, graph: networkx.DiGraph) -> Mapping[int, float]:
        raise NotImplementedError


class BetweennessCentralityHeuristic(CentralityHeuristic):
    def centrality(self, graph: networkx.DiGraph) -> Mapping[int, float]:
        return networkx.betweenness_centrality(G=graph)


class ClosenessCentralityHeuristic(CentralityHeuristic):
    def centrality(self, graph: networkx.DiGraph) -> Mapping[int, float]:
        return networkx.closeness_centrality(G=graph)


class DegreeCentralityHeuristic(CentralityHeuristic):
    def centrality(self, graph: networkx.DiGraph) -> Mapping[int, float]:
        return networkx.degree_centrality(G=graph)


class HarmonicCentralityHeuristic(CentralityHeuristic):
    def centrality(self, graph: networkx.DiGraph) -> Mapping[int, float]:
        return networkx.harmonic_centrality(G=graph)


class PageRankCentralityHeuristic(CentralityHeuristic):
    def centrality(self, graph: networkx.DiGraph) -> Mapping[int, float]:
        return networkx.pagerank(G=graph)


def get_node_with_max_min_ref_distance(
    sources: Collection[int],
    candidates: Collection[int],
    distances: numpy.ndarray,
) -> Tuple[float, Optional[int]]:
    r"""
    Compute
    .. math::
        argmax_c (min_s d(s, c))
    """
    # no candidates?
    if len(candidates) == 0:
        return float('-inf'), None

    # no sources?
    if len(sources) == 0:
        return float('inf'), next(iter(candidates))

    candidates = list(candidates)
    sources = list(sources)
    distances_from_sources = distances[sources]
    dist_sources_to_candidates = distances_from_sources[:, candidates]
    dist_closest_source_to_candidates = dist_sources_to_candidates.min(axis=0)
    i = dist_closest_source_to_candidates.argmax()
    distance = dist_closest_source_to_candidates[i]
    candidate = candidates[i]
    return distance, candidate


def _compute_shortest_path_distances(graph: KnowledgeGraph) -> numpy.ndarray:
    # create adjacency matrix
    source, target = graph.edge_tensor_unique
    weight = graph.edge_weights
    m = sparse.coo_matrix((weight.cpu().numpy(), (source.cpu().numpy(), target.cpu().numpy())), shape=(graph.num_entities, graph.num_entities)).todense()

    # compute all-pair-shortest paths
    dist_matrix = floyd_warshall(csgraph=m, directed=True, return_predecessors=False, unweighted=False, overwrite=True)

    return dist_matrix


class MaximumShortestPathDistanceHeuristic(BufferedHeuristic[Mapping[MatchSideEnum, numpy.ndarray]]):
    """Proposes nodes based on the shortest path distance to aligned nodes."""

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        artifact_root: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
            artifact_root=artifact_root,
            **kwargs
        )

    def _load(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        artifact_root: str,
    ) -> Mapping[MatchSideEnum, numpy.ndarray]:
        distances = {}
        os.makedirs(artifact_root, exist_ok=True)
        for side, graph in dataset.graphs.items():
            artifact_path = path.join(artifact_root, f'{str(side.value)}.npz')
            if path.isfile(artifact_path):
                logging.info(f'Loading shortest path distances from {artifact_path}')
                distances[side] = numpy.load(file=artifact_path)
            else:
                logging.info(f'{artifact_path} not existing. Computing shortest paths.')
                dist = _compute_shortest_path_distances(graph=graph)
                logging.info(f'Saving artifacts to {artifact_path}.')
                numpy.save(file=artifact_path, arr=dist)
                distances[side] = dist
        assert distances.keys() == dataset.graphs.keys()
        return distances

    def propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        sources = {k: set(v) for k, v in oracle.positives.items()}
        available = {k: set(v) for k, v in oracle.restricted_available(restrict_to=restrict_to).items()}
        queries: List[Tuple[MatchSideEnum, int]] = []

        for _ in range(num):
            query = max(
                get_node_with_max_min_ref_distance(
                    sources=sources[side],
                    candidates=available[side],
                    distances=distances,
                ) + (side,) for side, distances in self.buffer.items())[:0:-1]

            # Fallback to random heuristic
            if query is None:
                query = RandomHeuristic.random_selection(oracle=oracle, num=1)[0]
            side, node_id = query

            # add chosen node as source node (aligned nodes)
            sources[side].add(node_id)

            # remove node from list of available nodes
            available[side].remove(node_id)

            # add chosen node to final selection
            queries.append(query)

        return queries
