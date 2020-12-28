# coding=utf-8
import logging
from typing import List, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional

from .similarity import Similarity
from ..data import MatchSideEnum, SIDES

_LOGGER = logging.getLogger(name=__name__)


class PairwiseLoss(nn.Module):
    # pylint: disable=arguments-differ
    def forward(self, similarities: torch.FloatTensor, true_indices: torch.LongTensor) -> torch.FloatTensor:
        """
        Compute loss.

        :param similarities: shape: (n, m)
        :param true_indices: shape (n,)
        """
        raise NotImplementedError


class MarginLoss(PairwiseLoss):
    def __init__(self, margin: float = 1.0, exact_loss_value: bool = False):
        super().__init__()
        self.margin = margin
        self.exact_loss_value = exact_loss_value

    def forward(self, similarities: torch.FloatTensor, true_indices: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        n_left, n_right = similarities.shape
        batch_indices = torch.arange(n_left, device=similarities.device)
        pos_sim = similarities[batch_indices, true_indices].unsqueeze(dim=1)
        # as pos_sim + margin - pos_sim = margin, there is no gradient for comparison of positives with positives
        # as there are n_right elements per row, with one positive, and (n_right-1) negatives, we need to subtract
        # (margin/n_right) to compensate for that in the loss value.
        # As this is a constant, the gradient is the same as if we would not add it, hence we only do it, if explicitly requested.
        loss_value = functional.relu(similarities + self.margin - pos_sim).mean()
        if self.exact_loss_value:
            loss_value = loss_value - (float(self.margin) / n_right)
        return loss_value


class MatchingLoss(nn.Module):
    """An API for graph matching losses."""

    #: The similarity
    similarity: Similarity

    def __init__(self, similarity: Similarity):
        super().__init__()
        self.similarity = similarity

    # pylint: disable=arguments-differ
    def forward(
        self,
        alignment: torch.LongTensor,
        representations: Mapping[MatchSideEnum, torch.FloatTensor] = None,
        candidates: Mapping[MatchSideEnum, torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        :param alignment: shape: (2, num_aligned)
        :param representations:
            side -> repr, where repr is a tensor of shape (num_nodes_side, dim)
        :param candidates:
            side -> candidates, where candidates is of shape (num_candidates_side,)
        """
        if candidates is None:
            candidates = {}

        return 0.5 * sum(
            self._one_side_matching_loss(
                source=representations[source_side],
                target=representations[target_side],
                alignment=this_alignment,
                source_candidates=candidates.get(source_side),
                target_candidates=candidates.get(target_side),
            )
            for (source_side, target_side), this_alignment in zip(
                zip(SIDES, reversed(SIDES)),
                [alignment, alignment.flip(0)]
            )
        )

    def _one_side_matching_loss(
        self,
        source: torch.FloatTensor,
        target: torch.FloatTensor,
        alignment: torch.LongTensor,
        source_candidates: Optional[List[int]] = None,
        target_candidates: Optional[List[int]] = None,
    ) -> torch.FloatTensor:
        """
        Compute the loss from selected nodes in source graph to the other graph.

        :param source: shape: (num_source, dim)
            Source node representations.
        :param target: shape: (num_target, dim)
            Target node representations.
        :param alignment: shape: (2, num_aligned)
            The alignment.
        :param source_candidates: shape: (num_source_excl,)
            Exclusive nodes in source.
        :param target_candidates: shape: (num_target_excl,)
            Exclusive nodes in target.
        """
        raise NotImplementedError


class FullMatchingLoss(MatchingLoss):
    #: The pairwise loss
    pairwise_loss: PairwiseLoss

    def __init__(
        self,
        similarity: Similarity,
        pairwise_loss: PairwiseLoss,
    ):
        super().__init__(similarity=similarity)
        self.pairwise_loss = pairwise_loss

    def _one_side_matching_loss(
        self,
        source: torch.FloatTensor,
        target: torch.FloatTensor,
        alignment: torch.LongTensor,
        source_candidates: Optional[torch.LongTensor] = None,
        target_candidates: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if source_candidates is not None or target_candidates is not None:
            _LOGGER.warning('Negative labels are currently not used')
        source_ind, target_ind = alignment
        sim_some_src_to_all_target = self.similarity(source[source_ind], target)
        source_to_target_loss = self.pairwise_loss(similarities=sim_some_src_to_all_target, true_indices=target_ind)
        return source_to_target_loss.mean()


def sample_exclusive(
    ind_pos: torch.LongTensor,
    num_negatives: int,
    max_ind: int,
    blacklist: Optional[List[int]] = None,
) -> torch.LongTensor:
    r"""
    Draws random samples from {0, ..., max_ind} \ neg_exclusive

    The negative samples fulfil:

    .. math::
        negative\_samples[b, i] \neq  ind\_pos[b] \land
        negative\_samples[b, i] \notin blacklist

    and are sampled uniformly across all allowed IDs.

    :param ind_pos: shape: (batch_size,)
        The indices serving as positive samples.
    :param num_negatives: int, positive
        The number of negative samples to draw.
    :param max_ind: int, positive,
        The maximum index (defining the interval [0, max_ind), where the samples can be drawn from).
    :param blacklist: optional
        A list of forbidden indices (shared across all batch members).

    :return: shape: (batch_size, num_negatives)
        The negative samples.
    """
    if blacklist is None:
        blacklist = []

    num_exclusive = len(blacklist)
    device = ind_pos.device

    # shape: (batch_size, num_neg)
    neg_ind = torch.randint(max_ind - 1 - num_exclusive, size=(ind_pos.shape[0], num_negatives), dtype=torch.long, device=device)

    # shape: (1, n_excl)
    sorted_neg_excl = torch.as_tensor(sorted(blacklist), dtype=torch.long, device=device).unsqueeze(dim=0)

    #: shape: (b, n_excl)
    sorted_neg_excl = sorted_neg_excl - (sorted_neg_excl >= ind_pos.unsqueeze(dim=1)).long()
    for e in range(num_exclusive):
        # shift if at least as large as exclusive
        mask = neg_ind >= sorted_neg_excl[:, e, None]
        neg_ind[mask] += 1

    # shift if equal to pos
    mask = neg_ind >= ind_pos.unsqueeze(dim=1)
    neg_ind[mask] += 1

    return neg_ind


def sample_from_candidates(
    candidates: torch.LongTensor,
    batch_size: int,
    num_negatives: int,
) -> torch.LongTensor:
    return candidates[torch.randint(candidates.shape[0], size=(batch_size, num_negatives), device=candidates.device)]


class SampledMatchingLoss(MatchingLoss):
    #: The pairwise loss
    pairwise_loss: PairwiseLoss

    #: The number of negative samples
    num_negatives: int

    def __init__(
        self,
        similarity: Similarity,
        pairwise_loss: PairwiseLoss,
        num_negatives: int = 1,
        self_adversarial_weighting: bool = False,
    ):
        super().__init__(similarity=similarity)
        self.pairwise_loss = pairwise_loss
        self.num_negatives = num_negatives
        self.self_adversarial_weighting = self_adversarial_weighting

    def _one_side_matching_loss(
        self,
        source: torch.FloatTensor,
        target: torch.FloatTensor,
        alignment: torch.LongTensor,
        source_candidates: Optional[torch.LongTensor] = None,
        target_candidates: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Split mapping
        source_ind, target_ind_pos = alignment

        # Extract representations, shape: (batch_size, dim)
        anchor = source[source_ind]

        # Positive scores
        # positive samples, shape: (batch_size, dim)
        pos = target[target_ind_pos]

        # scores: shape: (batch_size,)
        pos_scores = self.similarity.one_to_one(left=anchor, right=pos)

        # Negative samples in target graph, shape: (batch_size, num_negatives)
        batch_size = target_ind_pos.shape[0]
        if target_candidates is not None:
            neg_ind = sample_from_candidates(candidates=target_candidates, batch_size=batch_size, num_negatives=self.num_negatives)
        else:
            # shape: (batch_size, num_neg)
            n_target = target.shape[0]
            neg_ind = torch.randint(n_target, size=(batch_size, self.num_negatives), dtype=torch.long, device=target_ind_pos.device)

        # Negative scores
        # negative samples, shape: (batch_size, num_negatives, dim)
        # equivalent to: neg = target[neg_ind.view(-1)].view(alignment.shape[1], self.num_negatives, -1)
        neg = target[neg_ind]

        # scores: shape: (batch_size, num_negatives)
        neg_scores = self.similarity.one_to_one(left=anchor.unsqueeze(1), right=neg)

        # self-adversarial weighting as described in RotatE paper: https://arxiv.org/abs/1902.10197
        if self.self_adversarial_weighting:
            w = functional.softmax(neg_scores, dim=1).detach()
            neg_scores = w * neg_scores

        # Evaluate pair loss
        scores = torch.cat([pos_scores.unsqueeze(dim=-1), neg_scores], dim=-1)
        return self.pairwise_loss(similarities=scores, true_indices=torch.zeros_like(target_ind_pos)).mean()
