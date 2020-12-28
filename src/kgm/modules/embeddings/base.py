from typing import Any, Mapping, Optional

import torch
from torch import nn

from .init import NodeEmbeddingInitMethod, NodeEmbeddingInitializer, init_method_normalizer
from .norm import NodeEmbeddingNormalizationMethod, NodeEmbeddingNormalizer, norm_method_normalizer
from ...data import MatchSideEnum, SIDES
from ...utils.common import get_subclass_by_name, reduce_kwargs_for_method
from ...utils.torch_utils import get_device


class NodeEmbeddings(nn.Module):
    """NodeEmbeddings abstraction layer."""
    #: The node embeddings, actually a nn.ModuleDict
    # {side: embedding}
    _embeddings: Mapping[MatchSideEnum, nn.Embedding]

    # The initializer
    initializer: NodeEmbeddingInitializer

    #: The normalizer
    normalizer: NodeEmbeddingNormalizer

    def __init__(
        self,
        initializer: NodeEmbeddingInitializer,
        num_nodes: Mapping[MatchSideEnum, int],
        embedding_dim: int,
        trainable: bool = True,
        normalizer: Optional[NodeEmbeddingNormalizer] = None,
        device: Optional[torch.device] = None,
        dropout: Optional[float] = None,
        shared: bool = False,
    ):
        super().__init__()

        # Store embedding initialization method for re-initialization
        self.initializer = initializer

        # Bind normalizer
        self.normalizer = normalizer

        # Node embedding dropout
        if dropout is not None:
            dropout = nn.Dropout(p=dropout)
        self.dropout = dropout

        # Whether to share embeddings for different nodes in the graph
        self.shared = shared

        # Store num nodes
        self.num_nodes = num_nodes

        # Resolve device
        device = get_device(device=device)

        # Allocate embeddings
        if self.shared:
            num_nodes = {
                side: 1
                for side in self.num_nodes.keys()
            }
        self._embeddings = nn.ModuleDict({
            side: nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                _weight=torch.empty(num_embeddings, embedding_dim, device=device)
            )
            for side, num_embeddings in num_nodes.items()
        })

        # Set trainability
        for emb in self._embeddings.values():
            emb.weight.requires_grad_(trainable)

        # Initialize
        self.reset_parameters()

    def get_embedding(
        self,
        side: MatchSideEnum,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        emb = self._embeddings[side]
        if indices is None and self.shared:
            return emb.weight.repeat(self.num_nodes[side], 1)
        if indices is None:
            return emb.weight
        if self.shared:
            indices = torch.zeros_like(indices)
        return emb(indices)

    def embeddings(self, indices: Optional[torch.LongTensor] = None) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        # If no indices are provided, get all embeddings
        if indices is None:
            indices = {}

        # get embeddings
        return {
            side: self.get_embedding(side=side, indices=indices.get(side))
            for side in SIDES
        }

    # pylint: disable=arguments-differ
    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        embeddings = self.embeddings(indices=indices)

        # apply dropout if requested
        if self.dropout is not None:
            embeddings = {
                side: self.dropout(embedding)
                for side, embedding in embeddings.items()
            }

        # Apply normalization if requested
        if self.normalizer is not None:
            embeddings = self.normalizer.normalize(x=embeddings)

        return embeddings

    def reset_parameters(self) -> None:
        """
        Re-initializes the embeddings as specified by the embedding initializer
        """
        self.initializer.initialize_(self.embeddings())

    @property
    def device(self) -> torch.device:
        if len(self._embeddings) > 0:
            return next(iter(self._embeddings.values())).weight.device
        else:
            return torch.device('cpu')


def get_embedding(
    init: NodeEmbeddingInitMethod,
    num_nodes: Mapping[MatchSideEnum, int],
    embedding_dim: int,
    device: Optional[torch.device] = None,
    dropout: Optional[float] = None,
    trainable: bool = True,
    init_config: Optional[Mapping[str, Any]] = None,
    norm: NodeEmbeddingNormalizationMethod = NodeEmbeddingNormalizationMethod.none,
) -> NodeEmbeddings:
    # Build initializer
    init_class = get_subclass_by_name(
        base_class=NodeEmbeddingInitializer,
        name=init.value,
        normalizer=init_method_normalizer,
    )

    # get a list of parameter names from the target init function and drop all unavailable keys from the config dict
    initializer = init_class(**(reduce_kwargs_for_method(method=init_class.__init__, kwargs=init_config)))

    # Build normalizer
    norm_class = get_subclass_by_name(
        base_class=NodeEmbeddingNormalizer,
        name=norm.value,
        normalizer=norm_method_normalizer,
    )
    normalizer = norm_class()

    return NodeEmbeddings(
        initializer=initializer,
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,
        trainable=trainable,
        normalizer=normalizer,
        dropout=dropout,
        device=device,
    )
