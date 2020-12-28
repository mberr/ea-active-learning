# coding=utf-8
import enum
import math
from abc import ABC, abstractmethod
from typing import Mapping

import torch
from torch import nn

from ....data import MatchSideEnum
from ....utils.torch_utils import truncated_normal_


class NodeEmbeddingInitializer:
    """Initialization methods."""

    @abstractmethod
    def initialize_(self, embeddings: Mapping[MatchSideEnum, torch.FloatTensor]) -> None:
        """Initialize the embeddings in-place.

        :param embeddings:
            The node embeddings.
        """
        raise NotImplementedError


class RandomNodeEmbeddingInitializer(NodeEmbeddingInitializer):
    """Initialize nodes i.i.d. with random vectors drawn from the given distribution."""

    def __init__(
        self,
        random_distribution=nn.init.normal_,
    ):
        self.random_dist_ = random_distribution

    def _std(
        self,
        side: MatchSideEnum,
        num_nodes: Mapping[MatchSideEnum, int]
    ) -> float:
        """Return the std for one side."""
        return math.sqrt(1. / num_nodes[side])

    def initialize_(self, embeddings: Mapping[MatchSideEnum, torch.FloatTensor]) -> None:  # noqa: D102
        num_nodes = {
            side: embedding.shape[0]
            for side, embedding in embeddings.items()
        }
        for side, embedding in embeddings.items():
            self.random_dist_(embedding, std=self._std(side=side, num_nodes=num_nodes))


class GcnNodeEmbeddingInitializer(RandomNodeEmbeddingInitializer, ABC):
    """Groups embedding initializers used by GCNAlign."""

    def __init__(self):
        super().__init__(random_distribution=truncated_normal_)


class SqrtTotalNodeEmbeddingInitializer(GcnNodeEmbeddingInitializer):
    """Uses the standard std of 1 / (sum #nodes)."""

    def _std(self, side: MatchSideEnum, num_nodes: Mapping[MatchSideEnum, int]) -> float:  # noqa: D102
        return 1. / math.sqrt(sum(num_nodes.values()))


class SqrtIndividualNodeEmbeddingInitializer(GcnNodeEmbeddingInitializer):
    """Uses the default standard std of 1 / #nodes."""


class StdOneNodeEmbeddingInitializer(GcnNodeEmbeddingInitializer):
    """Uses the standard std of 1."""

    def _std(self, side: MatchSideEnum, num_nodes: Mapping[MatchSideEnum, int]) -> float:  # noqa: D102
        return 1.


def init_method_normalizer(name: str):
    """Normalizes the name of an initialization method."""
    return name.lower().replace('_', '').replace('nodeembeddinginitializer', '')


class NodeEmbeddingInitMethod(str, enum.Enum):
    """Enum for selecting how to initialize node embeddings."""
    #: Use random initialization
    random = 'random'

    #: standard normal distribution
    std_one = 'std_one'

    #: std = 1 / sqrt(sum_i n_nodes_i)
    sqrt_total = 'sqrt_total'

    #: std = 1 / sqrt(n_nodes_i)
    sqrt_individual = 'sqrt_individual'

    def __str__(self):
        return self.name
