# coding=utf-8
"""
API for models for KG matching.
"""
import enum
from abc import ABC, abstractmethod
from typing import Callable, Mapping, MutableMapping, Optional, Type

import torch
from frozendict import frozendict
from torch import nn

from ...data import MatchSideEnum
from ...utils.common import get_subclass_by_name
from ...utils.torch_utils import resolve_device_from_to_kwargs


class KGMatchingModel(nn.Module):
    # pylint: disable=arguments-differ
    @abstractmethod
    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        """Return embeddings for nodes on both sides.

        :param indices:
            If provided only return representations for these indices.

        :return: a mapping side -> representations
            where
            representations: shape: (num_nodes_on_side, embedding_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def set_edge_tensors_(self, edge_tensors: Mapping[MatchSideEnum, torch.LongTensor]) -> None:
        """(Re-)Set the edge tensors.

        :param edge_tensors:
            A dictionary side -> edge_tensor where

                edge_tensor: shape: (2, num_edges)

        :return: None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The model's device."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_nodes(self) -> Mapping[MatchSideEnum, int]:
        """The number of nodes per side."""
        raise NotImplementedError


# pylint: disable=abstract-method
class AbstractKGMatchingModel(KGMatchingModel, ABC):
    """Abstract matching model taking care of moving edge tensors."""
    #: The edge tensors
    edges: MutableMapping[MatchSideEnum, torch.LongTensor]

    def __init__(self, num_nodes: Mapping[MatchSideEnum, int]):
        super().__init__()
        self.edges = {}
        self._num_nodes = frozendict(num_nodes)

    @property
    def num_nodes(self) -> Mapping[MatchSideEnum, int]:
        return self._num_nodes

    def set_edge_tensors_(self, edge_tensors: Mapping[MatchSideEnum, torch.LongTensor]) -> None:  # noqa: D102
        # send edge tensors to device
        for side, edge_tensor in edge_tensors.items():
            self.edges[side] = edge_tensor.to(self.device)

    def to(self, *args, **kwargs):  # noqa: D102
        super().to(*args, **kwargs)
        self._to(device=resolve_device_from_to_kwargs(args, kwargs))
        return self

    def _to(self, device: torch.device) -> None:
        """Move tensors to device."""
        pass


class EdgeWeightsEnum(str, enum.Enum):
    """Which edge weights to use."""
    #: None
    none = 'none'

    #: Inverse in-degree -> sum of weights for incoming messages = 1
    inverse_in_degree = 'inverse_in_degree'


def get_matching_model_by_name(name: str, normalizer: Optional[Callable[[str], str]] = None) -> Type[KGMatchingModel]:
    if normalizer is None:
        normalizer = str.lower
    return get_subclass_by_name(base_class=KGMatchingModel, name=name, normalizer=normalizer, exclude={AbstractKGMatchingModel})
