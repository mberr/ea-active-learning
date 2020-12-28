# coding=utf-8
import unittest
from typing import Any, Callable, Mapping, Optional, Type

import numpy
import torch
from sklearn.metrics import pairwise
from torch.autograd import gradcheck

from kgm.modules.similarity import BoundInverseTransformation, CosineSimilarity, DistanceToSimilarity, DotProductSimilarity, LpSimilarity, NegativeTransformation, Similarity, SimilarityEnum, get_similarity, l1c

_RELATIVE_TOLERANCE = 1.0e-04


class _CommonTransformationTests:
    #: The number of distances
    dim = (5, 7)

    #: The distances
    distances: Optional[torch.FloatTensor] = None

    #: The transformation class
    transformation_cls: Type[DistanceToSimilarity]

    def setUp(self):
        self.distances = torch.rand(*self.dim, requires_grad=True)
        self.transformation = self.transformation_cls()

    def test_anti_monotonicity(self):
        similarity = self.transformation(self.distances)
        assert similarity.shape == self.distances.shape

        flat_distances = self.distances.view(-1)
        flat_similarities = similarity.view(-1)

        distance_comparison = flat_distances.unsqueeze(dim=0) > flat_distances.unsqueeze(dim=1)
        inverse_similarity_comparison = flat_similarities.unsqueeze(dim=0) <= flat_similarities.unsqueeze(dim=1)

        # (A => B) <=> (~A | (A & B) )
        assert (~distance_comparison | (distance_comparison & inverse_similarity_comparison)).all()


class BoundInverseTransformationTests(_CommonTransformationTests, unittest.TestCase):
    transformation_cls = BoundInverseTransformation


class NegativeTransformationTests(_CommonTransformationTests, unittest.TestCase):
    transformation_cls = NegativeTransformation


class _CommonTests:
    #: The vector dimension
    dimension: int = 2

    #: The number of vectors on the left side
    num_left = 13

    #: The number of vectors on the right side
    num_right = 7

    #: The left vectors
    left: Optional[torch.FloatTensor] = None

    #: The right vectors
    right: Optional[torch.FloatTensor] = None

    def setUp(self):
        self.left = torch.rand(self.num_left, self.dimension, requires_grad=True)
        self.right = torch.rand(self.num_right, self.dimension, requires_grad=True)


class _CommonSimilarityTests(_CommonTests):
    #: The similarity to test
    similarity: Similarity

    #: An optional callable operating on numpy arrays
    numpy_pairwise_metric: Optional[Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]] = None

    #: Optional key word arguments passed to numpy_pairwise_metric
    numpy_pairwise_kwargs: Optional[Mapping[str, Any]] = None

    def test_all_to_all(self):
        """Test whether all_to_all can be called."""
        sim = self.similarity.all_to_all(left=self.left, right=self.right)
        assert sim.shape == (self.num_left, self.num_right)

        metric = self.__class__.numpy_pairwise_metric
        if metric is not None:
            left_numpy = self.left.detach().cpu().numpy()
            right_numpy = self.right.detach().cpu().numpy()
            kwargs = self.numpy_pairwise_kwargs
            if kwargs is None:
                kwargs = {}
            sim_by_numpy = metric(left_numpy, right_numpy, **kwargs)
            if 'cdist' in str(metric):
                sim_by_numpy = self.similarity.transformation(torch.as_tensor(data=sim_by_numpy)).numpy()
            sim_to_numpy = sim.detach().cpu().numpy()
            numpy.testing.assert_allclose(sim_to_numpy, sim_by_numpy, rtol=_RELATIVE_TOLERANCE)

        fake_loss = torch.mean(sim)
        fake_loss.backward()

    def test_one_to_one(self):
        """Test whether one_to_one can be called."""
        sim = self.similarity.one_to_one(left=self.left, right=self.left)
        assert sim.shape == (self.num_left,)

        fake_loss = torch.mean(sim)
        fake_loss.backward()

    def test_coherence(self):
        """Test whether one_to_one and all_to_all compute the same values."""
        all_sim = self.similarity.all_to_all(left=self.left, right=self.right)

        # broadcast
        left_rep = self.left.repeat_interleave(repeats=self.num_right, dim=0)
        assert left_rep.shape == (self.num_right * self.num_left, self.dimension)
        right_rep = self.right.repeat(self.num_left, 1)
        assert right_rep.shape == (self.num_right * self.num_left, self.dimension)
        one_sim = self.similarity.one_to_one(left=left_rep, right=right_rep).view(self.num_left, self.num_right)

        assert torch.allclose(one_sim, all_sim)


class DotProductSimilarityTests(_CommonSimilarityTests, unittest.TestCase):
    similarity = DotProductSimilarity()
    numpy_pairwise_metric = pairwise.linear_kernel


class L2SimilarityTests(_CommonSimilarityTests, unittest.TestCase):
    similarity = LpSimilarity(p=2, transformation=BoundInverseTransformation())
    numpy_pairwise_metric = pairwise.distance.cdist
    numpy_pairwise_kwargs = {'p': 2, 'metric': 'minkowski'}


class L1SimilarityTests(_CommonSimilarityTests, unittest.TestCase):
    similarity = LpSimilarity(p=1, transformation=NegativeTransformation())
    numpy_pairwise_metric = pairwise.distance.cdist
    numpy_pairwise_kwargs = {'p': 1, 'metric': 'minkowski'}


class CosineSimilarityTests(_CommonSimilarityTests, unittest.TestCase):
    similarity = CosineSimilarity()
    numpy_pairwise_metric = pairwise.cosine_similarity


class L1CDistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x1 = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
        self.x2 = torch.randn(30, 20, dtype=torch.double, requires_grad=True)

    def test_forward(self):
        d = l1c(self.x1, self.x2)
        exp_d = torch.norm(self.x1[:, None, :] - self.x2[None, :, :], p=1, dim=-1)
        assert (d == exp_d).all()

    def test_backward_call(self):
        d = l1c(self.x1, self.x2)
        fake_loss = d.mean()
        fake_loss.backward()

    def test_backward_value(self):
        # gradcheck takes a tuple of tensors as input, check if your gradient
        # evaluated with these tensors are close enough to numerical
        # approximations and returns True if they all verify this condition.
        input = (self.x1, self.x2)
        test = gradcheck(l1c, input, eps=1e-8, atol=1e-4)
        print(test)


def test_get_similarity():
    """Test helper method to get similarity."""
    for similarity_enum in SimilarityEnum:
        similarity = get_similarity(similarity=similarity_enum)
