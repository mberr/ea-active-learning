# coding=utf-8
import random
import unittest

import torch

from base import GenericTest, TestTests
from kgm.data import SIDES
from kgm.modules.losses import FullMatchingLoss, MarginLoss, MatchingLoss, PairwiseLoss, SampledMatchingLoss, sample_exclusive
from kgm.modules.similarity import DotProductSimilarity


class _PairwiseLossTests(GenericTest[PairwiseLoss]):
    #: The number of nodes on the left side
    num_left: int = 3

    #: The number of nodes on the right side
    num_right: int = 5

    def test_forward(self):
        similarities = torch.rand(self.num_left, self.num_right, requires_grad=True)
        true_indices = torch.randint(self.num_right, size=(self.num_left,), requires_grad=False)
        assert true_indices.shape == (similarities.shape[0],)
        assert true_indices.max() < similarities.shape[1]

        # forward pass
        loss_value = self.instance.forward(similarities=similarities, true_indices=true_indices)

        # backward pass
        loss_value.backward()


class MarginLossTests(_PairwiseLossTests, unittest.TestCase):
    cls = MarginLoss


class PairwiseLossTestTests(TestTests[PairwiseLoss], unittest.TestCase):
    base_cls = PairwiseLoss
    base_test_cls = _PairwiseLossTests


class _MatchingLossTests(GenericTest[MatchingLoss]):
    #: The dimensionality of the representations
    dim: int = 7

    #: The number of nodes on the left side
    num_left: int = 3

    #: The number of nodes on the right side
    num_right: int = 5

    #: The number of aligned nodes
    num_align: int = 2

    def setUp(self) -> None:
        if self.kwargs is None:
            self.kwargs = {}
        self.kwargs['similarity'] = DotProductSimilarity()
        super().setUp()

    def test_forward(self):
        representations = {
            side: torch.rand(num, self.dim, requires_grad=True)
            for side, num in zip(SIDES, [self.num_left, self.num_right])
        }
        alignment = torch.stack([torch.arange(self.num_align), torch.randperm(self.num_align)], dim=0)

        # forward pass
        loss_value = self.instance.forward(alignment=alignment, representations=representations, candidates=None)

        # backward pass
        loss_value.backward()


class FullMatchingLossTests(_MatchingLossTests, unittest.TestCase):
    cls = FullMatchingLoss
    kwargs = {'pairwise_loss': MarginLoss()}


class SampledMatchingLossTests(_MatchingLossTests, unittest.TestCase):
    cls = SampledMatchingLoss
    kwargs = {'pairwise_loss': MarginLoss(), 'num_negatives': 5}


class SelfAdversarialSampledMatchingLossTests(_MatchingLossTests, unittest.TestCase):
    cls = SampledMatchingLoss
    kwargs = {'pairwise_loss': MarginLoss(), 'num_negatives': 5, 'self_adversarial_weighting': True}


class MatchingLossTestTests(TestTests[MatchingLoss], unittest.TestCase):
    base_cls = MatchingLoss
    base_test_cls = _MatchingLossTests


def test_sample_exclusive():
    max_ind = 128
    batch_size = 5
    num_negatives = 3
    num_excl = 7

    ind_pos = torch.randint(max_ind, size=(batch_size,))

    positives = set(map(int, ind_pos))
    neg_exclusive = set()
    while len(neg_exclusive) < num_excl:
        neg = random.randrange(max_ind)
        if neg not in positives:
            neg_exclusive.add(neg)
    neg_exclusive = list(neg_exclusive)
    neg_ind = sample_exclusive(ind_pos=ind_pos, num_negatives=num_negatives, max_ind=max_ind, blacklist=neg_exclusive)

    assert neg_ind.shape == (batch_size, num_negatives)
    assert (0 <= neg_ind).all()
    assert (neg_ind < max_ind).all()

    neg_exclusive_tensor = torch.as_tensor(data=neg_exclusive, dtype=torch.long)
    for b in range(batch_size):
        # not equal to positive
        assert not (neg_ind[b] == ind_pos[b]).any()

        # not equal to any of the excluded
        assert not (neg_ind[b].unsqueeze(dim=0) == neg_exclusive_tensor.unsqueeze(dim=-1)).any()
