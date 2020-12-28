import unittest
from typing import Mapping

import torch

from base import GenericTest, TestTests
from kgm.data import KnowledgeGraphAlignmentDataset, get_erdos_renyi
from kgm.data.knowledge_graph import sub_graph_alignment
from kgm.models.matching.base import MatchSideEnum
from kgm.modules.embeddings import NodeEmbeddings
from kgm.modules.embeddings.init import NodeEmbeddingInitializer
from kgm.modules.embeddings.init.base import GcnNodeEmbeddingInitializer, RandomNodeEmbeddingInitializer, SqrtIndividualNodeEmbeddingInitializer, SqrtTotalNodeEmbeddingInitializer, StdOneNodeEmbeddingInitializer
from kgm.modules.embeddings.norm import L1NodeEmbeddingNormalizer, L2NodeEmbeddingNormalizer, LpNormalization, NodeEmbeddingNormalizer, NoneNodeEmbeddingNormalizer
from kgm.utils.common import kwargs_or_empty


class NodeEmbeddingTests(GenericTest[NodeEmbeddings], unittest.TestCase):
    embedding_dim: int = 7
    num_left: int = 3
    num_right: int = 5
    cls = NodeEmbeddings
    kwargs = dict(
        initializer=StdOneNodeEmbeddingInitializer(),
    )

    def setUp(self) -> None:
        self.kwargs = dict(kwargs_or_empty(self.kwargs))
        self.kwargs['embedding_dim'] = self.embedding_dim
        self.kwargs['num_nodes'] = {
            MatchSideEnum.left: self.num_left,
            MatchSideEnum.right: self.num_right,
        }
        super().setUp()

    def test_get_embedding(self):
        for side, num_nodes in self.instance.num_nodes.items():
            for indices in (None, torch.randperm(num_nodes)[:num_nodes // 2]):
                embedding = self.instance.get_embedding(side=side, indices=indices)

                # check for shape
                exp_num = num_nodes if indices is None else indices.shape[0]
                assert embedding.shape == (exp_num, self.embedding_dim)

                # check for content
                if indices is not None:
                    for i, ind in enumerate(indices):
                        assert (embedding[i] == self.instance._embeddings[side](ind)).all()


class _CommonEmbeddingInitializerTests:
    #: The number of entities
    num_entities: int = 33

    #: The embedding dimensionality
    embedding_dim: int = 3

    #: The dataset
    dataset: KnowledgeGraphAlignmentDataset

    #: The embeddings
    embeddings: Mapping[MatchSideEnum, torch.FloatTensor]

    def setUp(self) -> None:
        super().setUp()
        self.dataset = sub_graph_alignment(graph=get_erdos_renyi(num_entities=self.num_entities, num_relations=2, p=.3))
        self.embeddings = {
            side: torch.rand(num_nodes, self.embedding_dim)
            for side, num_nodes in self.dataset.num_nodes.items()
        }


class _NodeEmbeddingInitializerTests(_CommonEmbeddingInitializerTests, GenericTest[NodeEmbeddingInitializer]):

    def test_initialize_(self):
        old_tensor_id = {
            side: id(emb)
            for side, emb in self.embeddings.items()
        }
        old_tensor_data = {
            side: emb.detach().clone()
            for side, emb in self.embeddings.items()
        }

        self.instance.initialize_(embeddings=self.embeddings)

        for side, emb in self.embeddings.items():
            # check the object stayed the same, i.e. in-place modification
            assert id(emb) == old_tensor_id[side]

            # check new data
            assert not (old_tensor_data[side] == emb).all()


class RandomNodeInitializationTests(_NodeEmbeddingInitializerTests, unittest.TestCase):
    cls = RandomNodeEmbeddingInitializer


class SqrtTotalNodeEmbeddingInitializerTests(_NodeEmbeddingInitializerTests, unittest.TestCase):
    cls = SqrtTotalNodeEmbeddingInitializer


class SqrtIndividualNodeEmbeddingInitializerTests(_NodeEmbeddingInitializerTests, unittest.TestCase):
    cls = SqrtIndividualNodeEmbeddingInitializer


class StdOneNodeEmbeddingInitializerTests(_NodeEmbeddingInitializerTests, unittest.TestCase):
    cls = StdOneNodeEmbeddingInitializer


class NodeInitializerTestTests(TestTests[NodeEmbeddingInitializer], unittest.TestCase):
    base_cls = NodeEmbeddingInitializer
    base_test_cls = _NodeEmbeddingInitializerTests
    skip_cls = frozenset([GcnNodeEmbeddingInitializer])


class _NodeEmbeddingNormalizerTests(_CommonEmbeddingInitializerTests, GenericTest[NodeEmbeddingNormalizer]):
    batch_size: int = 7

    def test_normalize(self):
        embeddings = self.embeddings
        x_n = self.instance.normalize(x=embeddings)
        if self.cls != NoneNodeEmbeddingNormalizer:
            assert id(embeddings) != id(x_n)


class NoneNodeEmbeddingNormalizerTests(_NodeEmbeddingNormalizerTests, unittest.TestCase):
    cls = NoneNodeEmbeddingNormalizer


class L1NodeEmbeddingNormalizerTests(_NodeEmbeddingNormalizerTests, unittest.TestCase):
    cls = L1NodeEmbeddingNormalizer


class L2NodeEmbeddingNormalizerTests(_NodeEmbeddingNormalizerTests, unittest.TestCase):
    cls = L2NodeEmbeddingNormalizer


class NodeEmbeddingNormalizerTestsTests(TestTests[NodeEmbeddingNormalizer], unittest.TestCase):
    base_cls = NodeEmbeddingNormalizer
    base_test_cls = _NodeEmbeddingNormalizerTests
    skip_cls = frozenset([LpNormalization])
