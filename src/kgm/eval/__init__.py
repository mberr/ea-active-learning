# coding=utf-8
"""
Evaluation methods.
"""
from .common import get_rank
from .matching import evaluate_alignment, evaluate_model

__all__ = [
    'evaluate_alignment',
    'evaluate_model',
    'get_rank',
]
