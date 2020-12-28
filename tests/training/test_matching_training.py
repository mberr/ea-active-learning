# coding=utf-8
import unittest
from typing import Mapping, Optional

import torch

from kgm.data import MatchSideEnum, exact_self_alignment, get_erdos_renyi
from kgm.models import KGMatchingModel
from kgm.modules import DotProductSimilarity, MarginLoss
from kgm.modules.embeddings.base import get_embedding
from kgm.modules.embeddings.init.base import NodeEmbeddingInitMethod
from kgm.training.matching import AlignmentModelTrainer


class DummyModel(KGMatchingModel):
    def __init__(
        self,
        num_nodes: Mapping[MatchSideEnum, int],
        embedding_dim: int,
    ):
        super().__init__()
        self._num_nodes = num_nodes
        self.embeddings = get_embedding(init=NodeEmbeddingInitMethod.random, num_nodes=num_nodes, embedding_dim=embedding_dim)

    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        result = {
            side: wrapper.weight for side, wrapper in self.embeddings._embeddings.items()
        }
        if indices is not None:
            result = {
                side: emb.index_select(dim=0, index=indices)
                for side, emb in result.items()
            }
        return result

    def set_edge_tensors_(self, edge_tensors: Mapping[MatchSideEnum, torch.LongTensor]) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return self.embeddings.device

    @property
    def num_nodes(self) -> Mapping[MatchSideEnum, int]:
        return self._num_nodes


class AlignmentModelTrainerTests(unittest.TestCase):
    #: The number of entities
    num_entities: int = 7

    #: The number of relations
    num_relations: int = 5

    #: The embedding dimensionality
    embedding_dim: int = 3

    def setUp(self) -> None:
        graph = get_erdos_renyi(num_entities=self.num_entities, num_relations=self.num_relations, p=.5)
        dataset = exact_self_alignment(graph=graph, train_percentage=0.8).validation_split(train_ratio=0.8)
        self.dataset = dataset
        self.model = DummyModel(num_nodes=dataset.num_nodes, embedding_dim=self.embedding_dim)
        self.trainer = AlignmentModelTrainer(
            model=self.model,
            similarity=DotProductSimilarity(),
            loss_kwargs={'pairwise_loss': MarginLoss(margin=3.)}
        )

    def test_test(self):
        for result in self.trainer.train(
            edge_tensors={},
            train_alignment=self.dataset.alignment.train,
            exclusives=None,
            validation_alignment=self.dataset.alignment.validation,
        ):
            print(result)
