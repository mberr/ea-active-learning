# coding=utf-8
from collections import deque
from typing import Collection, Deque, List, Mapping, Optional, Tuple

import torch
from torch.distributions import Normal

from . import AlignmentOracle, DegreeCentralityHeuristic, ModelBasedHeuristic, NodeActiveLearningHeuristic
from ..data import KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES, get_other_side
from ..models import KGMatchingModel
from ..modules import Similarity
from ..utils.torch_utils import csls

__all__ = [
    'PreviousExperienceBasedHeuristic',
]


class PreviousExperienceBasedHeuristic(ModelBasedHeuristic):
    """A node active learning heuristic using the model output for decision."""

    history: Deque[List[Tuple[float, MatchSideEnum, int]]]

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        model: KGMatchingModel,
        similarity: Similarity,
        base_heuristic: Optional[NodeActiveLearningHeuristic] = None,
        prob_diff_threshold: float = 0.5,
        steps_to_look_back: Optional[int] = 3,
        csls_k: Optional[int] = 2,
        **kwargs
    ):
        super().__init__(model=model, similarity=similarity, **kwargs)
        if base_heuristic is None:
            base_heuristic = DegreeCentralityHeuristic(
                dataset=dataset,
                **kwargs
            )
        self.base_heuristic = base_heuristic
        self.csls_k = csls_k
        self.prob_difference_threshold = prob_diff_threshold
        self.history = deque(maxlen=steps_to_look_back)

    def _propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        # get representations
        node_repr = self.model.forward()

        # collect data using base heuristic
        selection = []
        if len(self.history) > 1:
            selection = self.add_prob_diff_based(
                node_repr=node_repr,
                oracle=oracle,
                restrict_to=restrict_to,
            )
        if len(selection) >= num:
            selection = sorted(selection, reverse=True)[:num]
        else:
            # Get remaining candidates by base heuristic
            base_selection = self.base_heuristic.propose_next_nodes(oracle=oracle, num=num - len(selection), restrict_to=restrict_to)

            device = next(iter(node_repr.values())).device
            for side in SIDES:
                # select indices on side
                indices_this_side = torch.as_tensor(data=[node_id for (_side, node_id) in base_selection if _side == side], dtype=torch.long, device=device)
                if indices_this_side.numel() == 0:
                    continue

                # compute similarities & update selection
                selection.extend(
                    (score, side, index)
                    for score, index in zip(
                        self.this_side_to_other_side_sim(
                            indices_this_side=indices_this_side,
                            node_repr=node_repr,
                            side=side,
                        ).max(dim=1).values.tolist(),
                        indices_this_side.tolist())
                )

        # save requests for next time
        self.history.append(selection)

        return [s[1:] for s in selection]

    def _fit_distribution(
        self,
        node_ids: Mapping[MatchSideEnum, Collection[int]],
        device: torch.device,
    ) -> Optional[Normal]:
        similarities = torch.as_tensor([
            sim_value
            for selection in self.history
            for (sim_value, side, node_id) in selection
            if node_id in node_ids[side]
        ], dtype=torch.float)

        if similarities.numel() == 0:
            return None

        # update distribution parameters based on the recently acquired labels
        mean = similarities.mean().detach().to(device=device)
        std = similarities.std().detach().to(device=device)
        return Normal(loc=mean, scale=std)

    def add_prob_diff_based(
        self,
        node_repr: Mapping[MatchSideEnum, torch.FloatTensor],
        oracle: AlignmentOracle,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[float, MatchSideEnum, int]]:
        """
        Computes two probability distributions based on max similarity to nodes in counterpart graph for candidate nodes selected in previous
        requests. One distribution describes the max similarity of nodes with matchings in counterpart graph and another of nodes without matchings.
        Given these distributionsб the probability of candidates to have a match based on the max similarity to nodes in counterpart graphs is estimated.
        The difference between estimated probability to have a match and be without match is computed. Candidates with this difference above threshold are
        returned

        :param node_repr:
        :param oracle:
        :return: Candidates with difference of estimated probabilities above threshold are
        returned
        """
        device = next(iter(node_repr.values())).device

        # extract similarities from recently labeled nodes
        # update distribution parameters based on the recently acquired labels
        positive_distribution = self._fit_distribution(node_ids=oracle.positives, device=device)
        if positive_distribution is None:
            return []
        negative_distribution = self._fit_distribution(node_ids=oracle.negatives, device=device)
        if negative_distribution is None:
            return []

        # score nodes
        result = []
        for side, indices in oracle.restricted_available(restrict_to=restrict_to).items():
            if restrict_to is not None:
                indices = indices.intersection(restrict_to)
            indices = torch.as_tensor(data=sorted(indices), dtype=torch.long, device=device)
            if indices.numel() == 0:
                continue

            # compute similarities
            this_to_all_other_sim = self.this_side_to_other_side_sim(
                indices_this_side=indices,
                node_repr=node_repr,
                side=side,
            )

            # the assumption is that max values are decisive to determine whether a node has a matching
            max_scores = this_to_all_other_sim.max(dim=1).values
            scores = negative_distribution.cdf(max_scores) - (1 - positive_distribution.cdf(max_scores))

            # Select those where the probability difference is above threshold
            selection_mask = scores > self.prob_difference_threshold
            scores = scores[selection_mask].tolist()
            indices = indices[selection_mask].tolist()
            result.extend((score, side, index) for score, index in zip(scores, indices))
        return result

    def this_side_to_other_side_sim(
        self,
        indices_this_side: torch.LongTensor,
        node_repr: Mapping[MatchSideEnum, torch.FloatTensor],
        side: MatchSideEnum,
    ) -> torch.FloatTensor:
        """
        Compute similarity between selected indices on one side to all nodes on the other side.

        :param indices_this_side: len: n_indices
            The selected indices.
        :param node_repr:
            The node representations for both sides.
        :param side:
            The side.

        :return: (n_indices, n_other_side)
            The similarity matrix.
        """
        return csls(
            sim=self.similarity.all_to_all(
                left=node_repr[side][indices_this_side],
                right=node_repr[get_other_side(side=side)]
            ),
            k=self.csls_k
        )
