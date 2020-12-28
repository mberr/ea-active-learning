import unittest
from typing import Any, Mapping, Optional, Type

import torch

from kgm.data import KnowledgeGraphAlignmentDataset, get_erdos_renyi
from kgm.data.knowledge_graph import sub_graph_alignment
from kgm.models import GCNAlign, get_matching_model_by_name
from kgm.models.matching.base import KGMatchingModel
from kgm.utils.common import kwargs_or_empty


def _random_tensor(num_nodes: int, num_edges: int) -> torch.LongTensor:
    return torch.randint(num_nodes, size=(2, num_edges))


def _get_cycle_edge_tensor(num_nodes: int) -> torch.LongTensor:
    source = torch.arange(num_nodes)
    target = torch.cat([source[-1:], source[:-1]], dim=0)
    return torch.stack([source, target], dim=0)


class _KGMatchingTests:
    num_entities: int = 33
    num_relations: int = 2
    embedding_dim: int = 7

    model_cls: Type[KGMatchingModel]
    model_kwargs: Optional[Mapping[str, Any]] = None
    model: KGMatchingModel
    dataset: KnowledgeGraphAlignmentDataset

    def setUp(self) -> None:
        self.dataset = sub_graph_alignment(
            graph=get_erdos_renyi(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                p=.5,
            ),
        )
        self.model = self.model_cls(
            num_nodes=self.dataset.num_nodes,
            embedding_dim=self.embedding_dim,
            **(kwargs_or_empty(kwargs=self.model_kwargs))
        )

    def test_name_resolution(self):
        name = self.model_cls.__name__.lower()
        model_cls = get_matching_model_by_name(name=name)
        assert model_cls == self.model_cls

    def test_forward(self):
        self.model.set_edge_tensors_(edge_tensors=self.dataset.edge_tensors)
        enriched_embeddings = self.model()

        assert len(enriched_embeddings) == len(self.dataset.edge_tensors)
        for side, size in self.dataset.num_nodes.items():
            assert enriched_embeddings[side].shape == (size, self.embedding_dim)

    def test_to(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.set_edge_tensors_(edge_tensors=self.dataset.edge_tensors)

        # send to device
        model_on_device = self.model.to(device=device)
        assert model_on_device is not None

        # check that all attributes reside on device
        for p in self.model.parameters(recurse=True):
            assert p.device == device

    def test_reset_parameters(self):
        self.model.reset_parameters()


class GCNAlignTests(_KGMatchingTests, unittest.TestCase):
    model_cls = GCNAlign


if __name__ == '__main__':
    unittest.main()
