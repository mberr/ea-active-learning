# coding=utf-8
import enum
from abc import abstractmethod
from typing import Mapping

import torch
from torch.nn import functional

from ...data import MatchSideEnum


class NodeEmbeddingNormalizer:
    """Node embedding normalization."""

    @abstractmethod
    def normalize(
        self,
        x: Mapping[MatchSideEnum, torch.FloatTensor],
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        """Normalize a batch of embeddings, e.g. during forward pass.

        :param x: shape: (batch_size, dim)
            The tensor of embeddings.
        """
        raise NotImplementedError


class LpNormalization(NodeEmbeddingNormalizer):
    def __init__(self, p: int):
        self.p = p

    def normalize(self, x: Mapping[MatchSideEnum, torch.FloatTensor]) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        return {
            side: functional.normalize(xx, p=self.p, dim=-1)
            for side, xx in x.items()
        }


def norm_method_normalizer(name: str):
    """Normalizes the name of a normalization method."""
    return name.lower().replace('_', '').replace('nodeembeddingnormalizer', '')


class L2NodeEmbeddingNormalizer(LpNormalization):
    def __init__(self):
        super().__init__(p=2)


class L1NodeEmbeddingNormalizer(LpNormalization):
    def __init__(self):
        super().__init__(p=1)


class NoneNodeEmbeddingNormalizer(NodeEmbeddingNormalizer):
    def normalize(self, x: Mapping[MatchSideEnum, torch.FloatTensor]) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        return x


@enum.unique
class NodeEmbeddingNormalizationMethod(str, enum.Enum):
    none = 'none'
    l2 = 'l2'
    l1 = 'l1'
