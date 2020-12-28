# coding=utf-8
import logging
from typing import Mapping, Optional

import torch

from . import MatchSideEnum
from ..utils.torch_utils import filter_edges_by_nodes

__all__ = [
    'EdgeModifierHeuristic',
    'RemoveEdgesHeuristic',
]

_LOGGER = logging.getLogger(__name__)


class EdgeModifierHeuristic:
    """A heuristic of how to deal with edges when removing exclusive nodes."""

    def modify_edges(
        self,
        edge_tensors: Mapping[MatchSideEnum, torch.LongTensor],
        alignment: torch.LongTensor,
        exclusives: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Mapping[MatchSideEnum, torch.LongTensor]:
        raise NotImplementedError


class RemoveEdgesHeuristic(EdgeModifierHeuristic):
    """Remove edges going to an exclusive node."""

    def modify_edges(
        self,
        edge_tensors: Mapping[MatchSideEnum, torch.LongTensor],
        alignment: torch.LongTensor,
        exclusives: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Mapping[MatchSideEnum, torch.LongTensor]:  # noqa: D102
        if exclusives is None:
            _LOGGER.warning('No exclusives were provided.')
            return edge_tensors

        return {
            side: filter_edges_by_nodes(
                edge_tensor=edge_tensor,
                negatives=exclusives[side],
            )
            for side, edge_tensor in edge_tensors.items()
        }
