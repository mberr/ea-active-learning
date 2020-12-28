# coding=utf-8
import logging
import random
import tempfile
import unittest
from collections import defaultdict
from typing import Any, Mapping, Optional, Tuple, Type

import torch

from base import GenericTest, TestTests
from kgm.active_learning import AlignmentOracle, CoreSetHeuristic, NodeActiveLearningHeuristic
from kgm.active_learning.base import ModelBasedHeuristic, RandomHeuristic
from kgm.active_learning.bayesian import BALDHeuristic, BayesianHeuristic, BayesianSoftmaxEntropyHeuristic, VariationRatioHeuristic
from kgm.active_learning.graph import ApproximateVertexCoverHeuristic, BetweennessCentralityHeuristic, BufferedHeuristic, CentralityHeuristic, ClosenessCentralityHeuristic, DegreeCentralityHeuristic, HarmonicCentralityHeuristic, MaximumShortestPathDistanceHeuristic, PageRankCentralityHeuristic
from kgm.active_learning.learning import PreviousExperienceBasedHeuristic
from kgm.active_learning.similarity import BaseMinMaxSimilarityHeuristic, BatchOptimizedMaxSimilarityHeuristic, MaxSimilarityHeuristic, MinMaxSimilarityHeuristic, MostProbableMatchingInUnexploredRegionHeuristic, OneVsAllBinaryEntropyHeuristic, SimilarityBasedHeuristic, SoftmaxEntropyHeuristic, _cluster_node_representations_in_joint_space, _create_one_hot_available_per_cluster_matrix
from kgm.data import KnowledgeGraphAlignmentDataset, exact_self_alignment, get_erdos_renyi, validation_split
from kgm.data.edge_modifiers import EdgeModifierHeuristic, RemoveEdgesHeuristic
from kgm.data.knowledge_graph import MatchSideEnum, sub_graph_alignment
from kgm.models import KGMatchingModel
from kgm.modules import DotProductSimilarity
from kgm.modules.embeddings.base import get_embedding
from kgm.modules.embeddings.init.base import NodeEmbeddingInitMethod


def _get_dataset(num_entities: int) -> KnowledgeGraphAlignmentDataset:
    dataset = sub_graph_alignment(graph=get_erdos_renyi(num_entities=num_entities, num_relations=2, p=.3))
    dataset.alignment = validation_split(dataset.alignment, train_ratio=0.8)
    return dataset


class OracleTests(unittest.TestCase):
    num_entities: int = 128

    def setUp(self) -> None:
        self.dataset = _get_dataset(num_entities=self.num_entities)
        self.oracle = AlignmentOracle(dataset=self.dataset)

    def _label_some_nodes(self):
        for i in range(10):
            side = MatchSideEnum.left if i % 2 == 0 else MatchSideEnum.right
            node_id = next(iter(self.oracle.available[side]))
            self.oracle.label_node(node_id=node_id, side=side)

    def test_positives(self):
        # label some nodes
        self._label_some_nodes()

        positives = self.oracle.positives

        # check keys
        assert set(positives.keys()) == {MatchSideEnum.left, MatchSideEnum.right}

        # check consistency with num_positives
        assert self.oracle.num_aligned == {side: len(exclusives) for side, exclusives in positives.items()}

    def test_negatives(self):
        # label some nodes
        self._label_some_nodes()

        negatives = self.oracle.negatives

        # check keys
        assert set(negatives.keys()) == {MatchSideEnum.left, MatchSideEnum.right}

        # check consistency with num_negatives
        assert self.oracle.num_exclusives == {side: len(exclusives) for side, exclusives in negatives.items()}

    def test_alignment(self):
        # label some nodes
        self._label_some_nodes()

        alignment = self.oracle.alignment

        # check shape to be (2, ?)
        assert alignment.ndimension() == 2
        assert alignment.shape[0] == 2

        # check consistency with num_alignment_pairs
        assert alignment.shape[1] == self.oracle.num_alignment_pairs

    def test_label_node(self):
        while self.oracle.num_available > 0:
            side, available_nodes = next(filter(lambda t: len(t[1]) > 0, self.oracle.available.items()))
            node_id = next(iter(available_nodes))
            self.oracle.label_node(node_id=node_id, side=side)
        assert set(zip(*self.oracle.alignment.tolist())).symmetric_difference(zip(*self.dataset.alignment.train.tolist())) == set()


class DummyModel(KGMatchingModel):
    def __init__(self, num_nodes: Mapping[MatchSideEnum, int], dropout: bool = False):
        super().__init__()
        self.embeddings = get_embedding(
            init=NodeEmbeddingInitMethod.random,
            num_nodes=num_nodes,
            embedding_dim=3,
            dropout=0.2 if dropout else None,
            trainable=True,
        )
        self._num_nodes = num_nodes

    @property
    def num_nodes(self) -> Mapping[MatchSideEnum, int]:
        return self._num_nodes

    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        return self.embeddings.forward(indices=indices)

    def set_edge_tensors_(self, edge_tensors: Mapping[MatchSideEnum, torch.LongTensor]) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return next(e.device for e in self.embeddings.values())


class _NodeActiveLearningHeuristicTests(GenericTest[NodeActiveLearningHeuristic]):
    model_kwargs: Optional[Mapping[str, Any]] = None
    oracle: AlignmentOracle
    num_entities: int = 128
    artifact_root: Optional[tempfile.TemporaryDirectory]

    def setUp(self) -> None:
        # Log everything
        logging.basicConfig(level=logging.DEBUG)

        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        self.dataset = _get_dataset(num_entities=self.num_entities)
        kwargs['dataset'] = self.dataset
        if issubclass(self.cls, (ModelBasedHeuristic)):
            dropout = issubclass(self.cls, BayesianHeuristic)
            self.model = DummyModel(num_nodes=self.dataset.num_nodes, dropout=dropout)
            self.similarity = DotProductSimilarity()
            kwargs['model'] = self.model
            kwargs['similarity'] = self.similarity
        else:
            self.model = None
            self.similarity = None
        # correct init arguments for buffered heuristics
        if issubclass(self.cls, (BufferedHeuristic, PreviousExperienceBasedHeuristic)):
            self.artifact_root = tempfile.TemporaryDirectory()
            kwargs['artifact_root'] = self.artifact_root.name
        else:
            self.artifact_root = None
        self.instance = self.cls(**kwargs)
        self.oracle = AlignmentOracle(dataset=self.dataset)

    def tearDown(self) -> None:
        if self.artifact_root is not None:
            self.artifact_root.cleanup()

    def _verify_query(self, query: Tuple[MatchSideEnum, int]) -> bool:
        assert len(query) == 2
        side, node_id = query
        assert isinstance(side, MatchSideEnum)
        assert isinstance(node_id, int)
        assert 0 <= node_id < self.num_entities
        return True

    def test_propose_next_nodes(self):
        while self.oracle.num_available > 0:
            num = min(20, self.oracle.num_available)
            queries = self.instance.propose_next_nodes(oracle=self.oracle, num=num)
            for query in queries:
                self._verify_query(query=query)
            for query in queries:
                side, node_id = query
                if self.oracle.check_availability(side=side, node_id=node_id):
                    self.oracle.label_node(node_id=node_id, side=side)

    def test_propose_next_nodes_with_restriction(self):
        while self.oracle.num_available > 0:
            num = min(20, self.oracle.num_available)
            k = min(2 * num, self.oracle.num_available)
            print(num, self.oracle.num_available, len(self.oracle.available_pairs), k)
            restricted_pairs = random.sample(list(self.oracle.available_pairs), k)
            restriction = defaultdict(set)
            for side, node_id in restricted_pairs:
                restriction[side].add(node_id)
            queries = self.instance.propose_next_nodes(oracle=self.oracle, num=num, restrict_to=restriction)
            for query in queries:
                self._verify_query(query=query)
            for query in queries:
                side, node_id = query
                if self.oracle.check_availability(side=side, node_id=node_id):
                    self.oracle.label_node(node_id=node_id, side=side)


class RandomActiveLearningHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = RandomHeuristic


class ApproximateVertexCoverHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = ApproximateVertexCoverHeuristic


class BetweennessCentralityHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = BetweennessCentralityHeuristic


class ClosenessCentralityHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = ClosenessCentralityHeuristic


class DegreeCentralityHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = DegreeCentralityHeuristic


class HarmonicCentralityHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = HarmonicCentralityHeuristic


class PageRankCentralityHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = PageRankCentralityHeuristic


class MaximumShortestPathDistanceHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = MaximumShortestPathDistanceHeuristic


class MaxSimilarityHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = MaxSimilarityHeuristic


class OneVsAllBinaryEntropyHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = OneVsAllBinaryEntropyHeuristic


class BatchOptimizedMaxSimilarityHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = BatchOptimizedMaxSimilarityHeuristic


class CoreSetHeuristicHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = CoreSetHeuristic


class MostProbableMatchingsInUnexploredRegionsHeuristicTest(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = MostProbableMatchingInUnexploredRegionHeuristic

    def test_cluster_node_representations_in_joint_space(self):
        node_repr = self.model.forward()
        number_centroids = self.instance.number_centroids
        clusters = _cluster_node_representations_in_joint_space(
            node_repr=node_repr,
            similarity=self.similarity,
            number_centroids=number_centroids,
            num_iterations=self.instance.num_iterations,
            tolerance=self.instance.tolerance,
        )

        # check result
        for side, clusters_on_side in clusters.items():
            # check shape
            assert clusters_on_side.shape == (node_repr[side].shape[0], number_centroids)

            # check that each node belongs to exactly one cluster
            assert (clusters_on_side.sum(dim=1) == 1).all()

    def test_create_one_hot_available_per_cluster_matrix(self):
        number_centroids = self.instance.number_centroids

        # generate random cluster assignment matrices.
        clusters = {
            side: torch.randint(self.instance.number_centroids, size=(num_nodes_on_side,))[:, None] == torch.arange(number_centroids)[None, :]
            for side, num_nodes_on_side in self.model.num_nodes.items()
        }

        # restrict to available nodes
        availability_matrices = _create_one_hot_available_per_cluster_matrix(
            clusters=clusters,
            oracle=self.oracle,
            restrict_to=None,
        )

        # check result
        for side, available_on_side in availability_matrices.items():
            # check shape
            assert available_on_side.shape == (self.model.num_nodes[side], number_centroids)

            # check subset property
            clusters_on_side = clusters[side]
            assert (clusters_on_side >= available_on_side).all()


class MinMaxSimilarityHeuristicHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = MinMaxSimilarityHeuristic


class SoftmaxEntropyHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = SoftmaxEntropyHeuristic


class PreviousExperienceBasedHeuristicTests(_NodeActiveLearningHeuristicTests, unittest.TestCase):
    cls = PreviousExperienceBasedHeuristic


class _BayesianHeuristicTests(_NodeActiveLearningHeuristicTests):
    model_kwargs = {'node_embedding_dropout': 0.2}


class BayesianSoftmaxEntropyHeuristicTests(_BayesianHeuristicTests, unittest.TestCase):
    cls = BayesianSoftmaxEntropyHeuristic


class VariationRatioHeuristicTests(_BayesianHeuristicTests, unittest.TestCase):
    cls = VariationRatioHeuristic


class BALDHeuristicTests(_BayesianHeuristicTests, unittest.TestCase):
    cls = BALDHeuristic


class HeuristicTestTests(TestTests[NodeActiveLearningHeuristic], unittest.TestCase):
    base_cls = NodeActiveLearningHeuristic
    base_test_cls = _NodeActiveLearningHeuristicTests
    skip_cls = {
        BayesianHeuristic,
        CentralityHeuristic,
        ModelBasedHeuristic,
        SimilarityBasedHeuristic,
        BufferedHeuristic,
        BaseMinMaxSimilarityHeuristic,
    }


class _EdgeModifierHeuristicTests:
    #: The tested class
    cls: Type[EdgeModifierHeuristic]

    #: Optional keyword arguments for constructor
    kwargs: Optional[Mapping[str, Any]] = None

    #: The instance
    heuristic: EdgeModifierHeuristic

    #: The number of entities
    num_entities: int = 128

    #: The number of exclusive nodes
    num_exclusives: int = 16

    #: Whether there should be no edges to exclusive nodes
    isolate_exclusives: bool = True

    def setUp(self) -> None:
        kwargs = self.kwargs
        if kwargs is None:
            kwargs = {}
        self.heuristic = self.cls(**kwargs)

    def test_modify_edges(self):
        dataset = exact_self_alignment(graph=get_erdos_renyi(num_entities=self.num_entities, num_relations=2, p=.3), train_percentage=0.8)
        oracle = AlignmentOracle(dataset=dataset)
        new_edge_tensors = self.heuristic.modify_edges(edge_tensors=dataset.edge_tensors, alignment=oracle.alignment, exclusives=oracle.exclusives)

        for side, new_edge_tensor in new_edge_tensors.items():
            assert torch.is_tensor(new_edge_tensor)
            assert new_edge_tensor.ndimension() == 2
            assert new_edge_tensor.shape[0] == 2

            # check if valid node ids are used
            assert (0 <= new_edge_tensor).all()
            assert (new_edge_tensor <= self.num_entities).all()


class RemoveEdgesHeuristicTests(_EdgeModifierHeuristicTests, unittest.TestCase):
    cls = RemoveEdgesHeuristic
