# coding=utf-8
"""
Bayesian Heuristics using the Monte-Carlo dropout distribution over the model's output.
"""

import heapq
import logging
from typing import List, Mapping, Optional, Tuple, Collection

import torch
from torch import nn

from . import AlignmentOracle, ModelBasedHeuristic
from ..data import MatchSideEnum, SIDES, get_other_side
from ..models import KGMatchingModel
from ..modules import Similarity

__all__ = [
    'BALDHeuristic',
    'BayesianSoftmaxEntropyHeuristic',
    'VariationRatioHeuristic',
]

_LOGGER = logging.getLogger(__name__)


class BayesianAggregator:
    """An aggregator used to process the similarites from individual Monte-Carlo dropout runs."""

    def __init__(
        self,
        temperature: Optional[float] = None,
    ):
        self.buffer = dict()
        self.temperature = temperature

    def update(
        self,
        side: MatchSideEnum,
        sim: torch.FloatTensor,
    ) -> None:
        """
        Updates the aggregator.

        :param side: The side.
        :param sim: shape: (num_candidates, num_nodes)
        """
        if sim.numel() == 0:
            _LOGGER.warning(f'Received empty similarity tensor for side {side}.')
            return

        # softmax with temperature
        if self.temperature is not None:
            sim = sim / self.temperature
        p_match = torch.softmax(sim, dim=1)

        # Aggregate in buffer
        self.buffer.setdefault(side, {
            'sum_p': 0.,
            'count': 0
        })
        self.buffer[side]['sum_p'] += p_match
        self.buffer[side]['count'] += 1

    def finalize(self) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        """Compute node scores."""
        # Compute average similarity distribution across all recorded runs.
        for side in self.buffer.keys():
            mean_p = self.buffer[side]['sum_p'] / self.buffer[side]['count']
            self.buffer[side] = mean_p

        # Extract real results from average softmax distribution
        ret = self._finalize()  # pylint: disable=E1128

        # release buffers
        self.buffer.clear()

        # check whether the result was computed
        if ret is None:
            raise NotImplementedError

        return ret

    def _finalize(self) -> Optional[Mapping[MatchSideEnum, torch.FloatTensor]]:
        """The real work for finalization."""
        return None


class SoftmaxEntropyBayesianAggregator(BayesianAggregator):
    """Entropy of the average softmax distribution."""

    def _finalize(self) -> Optional[Mapping[MatchSideEnum, torch.FloatTensor]]:  # noqa: D102
        return {
            side: (mean_p.clamp_min(min=1.0 - 16).log() * mean_p).sum(dim=1)
            for side, mean_p in self.buffer.items()
        }


class VariationRatioAggregator(BayesianAggregator):
    """Maximise variation ratio: `1 - max p(c | x)` (Freeman, 1965)"""

    def _finalize(self) -> Mapping[MatchSideEnum, torch.FloatTensor]:  # noqa: D102
        return {
            side: 1. - mean_p.max(dim=1)[0]
            for side, mean_p in self.buffer.items()
        }


class BALDAggregator(BayesianAggregator):
    """Compute BALD."""

    def update(self, side: MatchSideEnum, sim: torch.FloatTensor) -> None:  # noqa: D102
        if sim.numel() == 0:
            _LOGGER.warning(f'Received empty similarity tensor for side {side}.')
            return

        # custom update; compute also average over entropies of individual distributions
        p = sim.softmax(dim=1)
        log_p = sim.log_softmax(dim=1)
        entropy = (p * log_p).sum(dim=1)

        self.buffer.setdefault(side, {
            'sum_p': 0.,
            'count': 0,
            'entropy_sum': 0.,
        })
        self.buffer[side]['sum_p'] += p
        self.buffer[side]['count'] += 1
        self.buffer[side]['entropy_sum'] += entropy

    def finalize(self) -> Mapping[MatchSideEnum, torch.FloatTensor]:  # noqa: D102
        # custom finalize; compute also average over entropies of individual distributions
        result = {}
        for side, buffer in self.buffer.items():
            count = buffer.pop('count')

            # compute entropy of the mean distribution
            x = buffer.pop('sum') / count
            x = x * x.log()
            x[torch.isnan(x)] = 0.
            x = -x.sum(dim=1)

            # subtract mean of the distributions' entropy
            result[side] = x + buffer.pop('entropy_sum') / count

        # release buffers
        self.buffer.clear()

        return result


class NormalEntropyAggregator(BayesianAggregator):
    """Use entropy of Normal distribution of similarity values."""

    def _reduce(self, entropy: torch.FloatTensor, dim: int) -> torch.FloatTensor:
        return entropy.max(dim=dim)

    def update(
        self,
        side: MatchSideEnum,
        sim: torch.FloatTensor,
    ) -> None:  # noqa: D102
        # Aggregate in buffer
        self.buffer.setdefault(side, {
            'sum_sim': 0.,
            'sum_sim2': 0.,
            'count': 0,
        })
        self.buffer[side]['sum_sim'] += sim
        self.buffer[side]['sum_sim2'] += sim ** 2
        self.buffer[side]['count'] += 1

    def finalize(self) -> Mapping[MatchSideEnum, torch.FloatTensor]:  # noqa: D102
        # Compute variance of similarity matrix, shape: (n_left, n_right)
        cnt = self.buffer.pop('count')
        x = (self.buffer.pop('sum_sim2') / cnt) - (self.buffer.pop('sum_sim') / cnt) ** 2

        # release buffers
        self.buffer.clear()

        # compute entropy of normal distribution with given variance
        #: entropy = 0.5 * log(2 * pi * e * var) = 0.5 * log(var) + 0.5 * log(2*pi*e) => prop to log(var)
        x = x.log()

        # Score: maximum entropy in similarity
        return {
            side: self._reduce(entropy=x, dim=dim)
            for dim, side in enumerate(SIDES)
        }


class BayesianHeuristic(ModelBasedHeuristic):
    """A node active learning heuristic using the Monte-Carlo dropout distribution over model outputs for decision."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        num_samples: Optional[int] = None,
        use_max_entropy: bool = True,
        **kwargs,
    ):
        super().__init__(model=model, similarity=similarity, **kwargs)
        if num_samples is None:
            num_samples = 25
        self.num_samples = num_samples
        self.sort_factor = 1 if use_max_entropy else -1

    @property
    def _dropout_layers(self) -> List[nn.Dropout]:
        """Get the dropout layers."""
        return [module for module in self.model.modules() if isinstance(module, nn.Dropout)]

    def get_aggregator(self) -> BayesianAggregator:
        """Instantiate an aggregator which processes the similarities for each individual Monte-Carlo dropout run."""
        raise NotImplementedError

    def _propose_next_nodes(
        self,
        oracle: AlignmentOracle,
        num: int,
        restrict_to: Optional[Mapping[MatchSideEnum, Collection[int]]] = None,
    ) -> List[Tuple[MatchSideEnum, int]]:  # noqa: D102
        if len(self._dropout_layers) == 0:
            raise AssertionError('Trying to use dropout-based output distribution, but model does not contain dropout.')

        # Set model into evaluation mode
        self.model.eval()

        # Enable training mode for all dropout layers
        for dropout in self._dropout_layers:
            dropout.train(True)

        # Ensure consistent order
        available = {
            side: sorted(indices)
            for side, indices in oracle.restricted_available(restrict_to=restrict_to).items()
        }

        aggregator = self.get_aggregator()
        for _ in range(self.num_samples):
            # Get node representations
            node_repr = self.model.forward()

            for side, indices in available.items():
                # process
                aggregator.update(
                    side=side,
                    sim=self.similarity.all_to_all(
                        left=node_repr[side][indices],
                        right=node_repr[get_other_side(side=side)],
                    )
                )

        # calculate node scores
        node_scores = aggregator.finalize()

        # check
        for side, indices in available.items():
            assert len(node_scores.get(side, [])) == len(indices)

        # Return items with best score
        return [
            tup[1:] for tup in heapq.nlargest(
                num,
                ((score, side, node_id)
                 for side, scores in node_scores.items()
                 for node_id, score in zip(available[side], scores.cpu().tolist())),
                key=lambda x: x[0] * self.sort_factor
            )
        ]


class BayesianSoftmaxEntropyHeuristic(BayesianHeuristic):
    """Ranks node according to the entropy of the mean softmax distribution over similarities to the other graph's nodes."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        *args,
        num_samples: Optional[int] = None,
        use_max_entropy: bool = True,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            *args,
            num_samples=num_samples,
            use_max_entropy=use_max_entropy,
            **kwargs,
        )
        self.temperature = temperature

    def get_aggregator(self) -> BayesianAggregator:  # noqa: D102
        return SoftmaxEntropyBayesianAggregator(temperature=self.temperature)


class VariationRatioHeuristic(BayesianHeuristic):
    """Selects nodes according to maximum variation ratio."""

    def get_aggregator(self) -> BayesianAggregator:  # noqa: D102
        return VariationRatioAggregator()


class BALDHeuristic(BayesianHeuristic):
    """Selects nodes according to maximum BALD."""

    def get_aggregator(self) -> BayesianAggregator:  # noqa: D102
        return BALDAggregator()
