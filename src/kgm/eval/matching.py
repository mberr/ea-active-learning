# coding=utf-8
from typing import Collection, Dict, Mapping, Optional, TypeVar

import torch

from .common import get_rank
from ..data import MatchSideEnum, SIDES
from ..models import KGMatchingModel
from ..modules import Similarity

__all__ = [
    'evaluate_model',
    'evaluate_alignment',
]

T = TypeVar('T')


def evaluate_model(
    model: KGMatchingModel,
    alignments: Mapping[T, torch.LongTensor],
    similarity: Similarity,
    eval_batch_size: int,
    ks: Collection[int] = (1, 10, 50, 100),
) -> Mapping[T, Mapping[str, float]]:
    """Evaluate a model on multiple alignments.

    :param model:
        The KG matching model to evaluate.
    :param alignments:
        A mapping of key -> alignment, where alignment is a LongTensor of shape (2, num_alignments).
    :param similarity:
        The similarity.
    :param eval_batch_size:
        The evaluation batch size.
    :param ks:
        The values for which to evaluate hits@k.

    :return:
        A mapping key -> subresult, where subresult is a mapping from metric-name to metric value.
    """
    # Evaluation
    with torch.no_grad():
        # Set model in evaluation mode
        model.eval()

        embeddings = model()

        result = {
            key: evaluate_alignment(similarity=similarity, alignment=alignment, representations=embeddings, eval_batch_size=eval_batch_size, ks=ks)
            for key, alignment in alignments.items()
        }
    return result


def evaluate_alignment(
    similarity: Similarity,
    alignment: torch.LongTensor,
    left: Optional[torch.FloatTensor] = None,
    right: Optional[torch.FloatTensor] = None,
    representations: Mapping[MatchSideEnum, torch.FloatTensor] = None,
    eval_batch_size: Optional[int] = None,
    ks: Collection[int] = (1, 10, 50, 100),
) -> Dict[str, float]:
    """
    Evaluate an alignment.

    :param left: shape: (n_l, d)
        The left node embeddings.
    :param right: shape: (n_r, d)
        The right node embeddings.
    :param alignment: shape: (2, a)
        The alignment.
    :param similarity:
        The similarity.
    :param eval_batch_size: int (positive, optional)
        The batch size to use for evaluation.
    :param ks:
        The values for which to compute hits@k.

    :return: A dictionary with keys 'mr, 'mrr', 'hits_at_k' for all k in ks.
    """
    num_alignments = alignment.shape[1]
    if left is not None or right is not None:
        raise AssertionError
    left, right = [representations[side] for side in SIDES]
    device = left.device
    alignment = alignment.to(device=device)
    all_left, all_right = alignment
    ranks: torch.FloatTensor = torch.empty(2, num_alignments, device=device)
    if eval_batch_size is None:
        eval_batch_size = num_alignments
    for i in range(0, num_alignments, eval_batch_size):
        left_ids, right_ids = alignment[:, i:i + eval_batch_size]
        num_match = left_ids.shape[0]
        true: torch.LongTensor = torch.arange(i, i + num_match, dtype=torch.long, device=device)

        sim_right_to_all_left = similarity.all_to_all(left[all_left], right[right_ids]).t()
        ranks[0, i:i + eval_batch_size] = get_rank(sim=sim_right_to_all_left, true=true)

        sim_left_to_all_right = similarity.all_to_all(left[left_ids], right[all_right])
        ranks[1, i:i + eval_batch_size] = get_rank(sim=sim_left_to_all_right, true=true)
    return_d = {
        'mr': torch.mean(ranks).item(),
        'amr': 2. * torch.mean(ranks).item() / (num_alignments + 1),
        'mrr': torch.mean(torch.reciprocal(ranks)).item(),
    }
    for k in ks:
        return_d[f'hits_at_{k}'] = torch.mean((ranks <= k).float()).item()
    return return_d
