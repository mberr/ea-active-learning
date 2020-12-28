# coding=utf-8
import enum
from abc import abstractmethod
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional

from kgm.utils.common import get_subclass_by_name, value_to_enum
from kgm.utils.torch_utils import csls


class DistanceToSimilarity(nn.Module):
    """A method to convert distances to similarities."""

    # pylint: disable=arguments-differ
    @abstractmethod
    def forward(self, distances: torch.FloatTensor) -> torch.FloatTensor:
        """
        Transforms a distance value to a similarity value.

        :param distances: The distances.

        :return: The similarities.
        """
        raise NotImplementedError


class BoundInverseTransformation(DistanceToSimilarity):
    r"""
    Compute the similarity using

    .. math::

            sim = \frac{1}{1 + dist}
    """

    def forward(self, distances: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return (distances + 1).reciprocal()


class NegativeTransformation(DistanceToSimilarity):
    r"""
        Compute the similarity using

        .. math::

                sim = -dist
        """

    def forward(self, distances: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        return -distances


class SimilarityEnum(str, enum.Enum):
    """How to determine node/relation similarity."""
    #: Dot product
    dot = 'dot'

    #: L2-distance based
    l2 = 'l2'

    #: L1-distance based
    l1 = 'l1'

    #: Cosine similarity
    cos = 'cos'


class Similarity(nn.Module):
    """
    Base class for similarity functions.
    """

    # pylint: disable=arguments-differ
    def forward(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute pairwise similarity scores.

        :param left: shape: (n, d)
        :param right: shape: (m, d)

        :return shape: (m, n)
        """
        return self.all_to_all(left=left, right=right)

    @abstractmethod
    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute pairwise similarity scores.

        .. math::

            out[i, j] = sim(left[i], right[j])

        :param left: shape: (n, d)
        :param right: shape: (m, d)

        :return shape: (m, n)
            sim_ij = sim(left_i, right_j)
        """
        raise NotImplementedError

    @abstractmethod
    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute similarity scores.

        .. math::

            out[i] = sim(left[i], right[i])

        :param left: shape: (n, d)
        :param right: shape: (n, d)

        :return shape: (n,)
        """
        raise NotImplementedError


class DotProductSimilarity(Similarity):
    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return left @ right.t()

    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return torch.sum(left * right, dim=-1)


class LpSimilarity(Similarity):
    def __init__(self, p: int = 2, transformation: DistanceToSimilarity = None):
        super().__init__()
        if transformation is None:
            transformation = BoundInverseTransformation()
        self.p = p
        self.transformation = transformation

    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        if self.p == 1:
            # work-around to avoid memory issue
            distances = l1c(left, right)
        elif self.p == 2:
            # work-around to avoid memory issue in backward pass, cf. https://github.com/pytorch/pytorch/issues/31599
            # || x - y ||**2 = <x-y, x-y> = <x,x> + <y,y> - 2<x,y>
            distances = ((left ** 2).sum(dim=-1).unsqueeze(dim=1) + (right ** 2).sum(dim=-1).unsqueeze(dim=0) - 2. * left @ right.t()).relu().sqrt()
        else:
            distances = torch.cdist(left, right, p=self.p)
        return self.transformation(distances)

    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.transformation(torch.norm(left - right, dim=-1, p=self.p))

    def __str__(self):
        return f'{self.__class__.__name__}(p={self.p}, transformation={self.transformation})'


class CosineSimilarity(Similarity):
    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        left_n = functional.normalize(left, p=2, dim=-1)
        right_n = functional.normalize(right, p=2, dim=-1)
        return left_n @ right_n.t()

    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        left_n = functional.normalize(left, p=2, dim=-1)
        right_n = functional.normalize(right, p=2, dim=-1)
        return (left_n * right_n).sum(dim=-1)


def transformation_normalizer(name: str) -> str:
    return name.lower().replace('transformation', '')


def get_similarity(
    similarity: Union[SimilarityEnum, str],
    transformation: Optional[Union[DistanceToSimilarity, str]] = None,
) -> Similarity:
    """
    Utility method to resolve similarity.

    :param similarity: The chosen similarity as enum.
    :param transformation: The transformation to use to convert distances to similarities.

    :return: The similarity function.
    """
    if not isinstance(similarity, SimilarityEnum):
        similarity = value_to_enum(enum_cls=SimilarityEnum, value=similarity)
    if isinstance(transformation, str):
        transformation = get_subclass_by_name(base_class=DistanceToSimilarity, name=transformation, normalizer=transformation_normalizer)()

    if similarity == SimilarityEnum.dot:
        return DotProductSimilarity()
    elif similarity == SimilarityEnum.l2:
        return LpSimilarity(p=2, transformation=transformation)
    elif similarity == SimilarityEnum.l1:
        return LpSimilarity(p=1, transformation=transformation)
    elif similarity == SimilarityEnum.cos:
        return CosineSimilarity()
    else:
        raise KeyError(f'Unknown similarity: {similarity}')


# TODO: Workaround until https://github.com/pytorch/pytorch/issues/24345 is fixed
# Inherit from Function
class L1CDist(torch.autograd.Function):
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, x1, x2):
        ctx.save_for_backward(x1, x2)

        # cdist.forward does not have the memory problem
        return torch.cdist(x1, x2, p=1)

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad_dist):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_x1 = grad_x2 = None

        # Retrieve saved values
        x1, x2 = ctx.saved_tensors
        dims = x1.shape[1]

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_x1 = torch.empty_like(x1)
        if ctx.needs_input_grad[1]:
            grad_x2 = torch.empty_like(x2)

        if any(ctx.needs_input_grad):
            for i in range(dims):
                #: sign: shape: (n1, n2)
                sign = torch.sign(x1[:, None, i] - x2[None, :, i])
                if ctx.needs_input_grad[0]:
                    grad_x1[:, i] = torch.sum(grad_dist * sign, dim=1)
                if ctx.needs_input_grad[1]:
                    grad_x2[:, i] = -torch.sum(grad_dist * sign, dim=0)

        return grad_x1, grad_x2


l1c = L1CDist.apply


class SimilarityNormalization(nn.Module):
    """A normalization method for similarity matrices."""

    def forward(self, similarity: torch.FloatTensor) -> torch.FloatTensor:
        """
        Normalize similarity.

        :param similarity: shape: (d1, d2)
            The similarity matrix.

        :return: shape: (d1, d2)
            The normalized similarity matrix.
        """
        raise NotImplementedError


class CSLSNormalization(SimilarityNormalization):
    """CSLS normalization."""

    def __init__(
        self,
        k: int = 1,
    ):
        super().__init__()
        self.k = k

    def forward(self, similarity: torch.FloatTensor) -> torch.FloatTensor:
        return csls(sim=similarity, k=self.k)


def generalized_k_means(
    x: torch.FloatTensor,
    similarity: Similarity,
    k: int,
    max_iter: int,
    tolerance: float,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Compute k-means style clusters by iteratively computing assignment to centers, and updating centers as the mean
    over members.

    :param x: shape: (n, d)
        The vectors to cluster.
    :param similarity:
        The similarity.
    :param k: > 0
        The number of clusters.
    :param max_iter: > 0
        The maximum number of iterations
    :param tolerance:
        A tolerance for convergence check.

    :return:
        A tuple (assignment, centers).
    """
    n, d = x.shape
    device = x.device

    # Randomly choose centers
    center_ind = torch.randperm(n, device=device)[:k]
    centers = x[center_ind]

    # Iterate
    for _ in range(max_iter):
        # Compute assignment
        #: shape: (c, n)
        center_x_similarity = similarity.all_to_all(left=x, right=centers)
        assignment = torch.argmax(center_x_similarity, dim=1)

        # Compute means
        new_centers = x.new_zeros(size=(k, d))
        new_centers.index_add_(dim=0, index=assignment, source=x)
        cluster_index, cluster_counts = torch.unique(assignment, sorted=False, return_counts=True)
        cluster_size = x.new_ones(size=(k,))
        cluster_size[cluster_index] = cluster_counts.float()
        new_centers = new_centers / cluster_size.unsqueeze(dim=1)

        # check for convergence
        # TODO: Use similarity instead
        if torch.norm(new_centers - centers, p=2, dim=-1).max().item() < tolerance:
            break

        centers = new_centers
    return assignment, centers
