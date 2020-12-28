# coding=utf-8
import logging
from typing import Mapping, Optional

import torch
from torch import nn

from .base import AbstractKGMatchingModel
from ...data import MatchSideEnum
from ...modules.embeddings import NodeEmbeddings
from ...modules.embeddings.base import get_embedding
from ...modules.embeddings.init.base import NodeEmbeddingInitMethod
from ...modules.embeddings.norm import NodeEmbeddingNormalizationMethod
from ...modules.graph import EdgeWeighting, GCNBlock, IdentityMessageCreator, InverseTargetInDegreeWeighting, MessagePassingBlock, OnlyUpdate, SumAggregator


class GCNAlign(AbstractKGMatchingModel):
    #: The node embeddings
    node_embeddings: NodeEmbeddings

    #: The edge weights
    edge_weights: Mapping[MatchSideEnum, torch.FloatTensor]

    def __init__(
        self,
        num_nodes: Mapping[MatchSideEnum, int] = None,
        embedding_dim: int = 200,
        device: torch.device = torch.device('cpu'),
        activation_cls: nn.Module = nn.ReLU,
        n_layers: int = 2,
        edge_weighting: Optional[EdgeWeighting] = None,
        use_conv_weights: bool = False,
        node_embedding_init_method: NodeEmbeddingInitMethod = NodeEmbeddingInitMethod.sqrt_total,  # 'total',  # 'individual'
        vertical_sharing: bool = True,
        node_embedding_dropout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(num_nodes=num_nodes)
        if len(kwargs) > 0:
            logging.warning(f'ignored kwargs: {kwargs}')

        # edge weighting
        if edge_weighting is None:
            edge_weighting = InverseTargetInDegreeWeighting()
        self.edge_weighting = edge_weighting
        logging.info(f'Using edge weighting {edge_weighting}.')
        self.edge_weights = {}

        # node embeddings
        self.node_embeddings = get_embedding(
            init=node_embedding_init_method,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            device=device,
            dropout=node_embedding_dropout,
            trainable=True,
            norm=NodeEmbeddingNormalizationMethod.l2,
        )

        # GCN layers
        self.n_layers = n_layers
        self.use_conv_weights = use_conv_weights
        self.vertical_sharing = vertical_sharing
        blocks = []
        if use_conv_weights:
            if self.vertical_sharing:
                gcn_block = GCNBlock(in_features=embedding_dim, out_features=embedding_dim, use_bias=True)
                activation = activation_cls()
                for i in range(n_layers):
                    blocks.append(gcn_block)
                    blocks.append(activation)
            else:
                for i in range(n_layers):
                    gcn_block = GCNBlock(in_features=embedding_dim, out_features=embedding_dim, use_bias=True)
                    activation = activation_cls()
                    blocks.append(gcn_block)
                    blocks.append(activation)
        else:
            message_block = MessagePassingBlock(
                message_creator=IdentityMessageCreator(),
                message_aggregator=SumAggregator(),
                node_updater=OnlyUpdate(),
            )
            for i in range(n_layers):
                blocks.append(message_block)
                activation = activation_cls()
                blocks.append(activation)
        side_to_modules = {
            side: nn.ModuleList(blocks)
            for side in num_nodes.keys()
        }
        self.layers = nn.ModuleDict(modules=side_to_modules)

        # Initialize parameters
        self.reset_parameters()

    def _to(self, device):  # noqa: D102
        super()._to(device=device)

        self.edge_weights = {
            side: weights.to(device)
            for side, weights in self.edge_weights.items()
        }

        return self

    def set_edge_tensors_(self, edge_tensors: Mapping[MatchSideEnum, torch.LongTensor]) -> None:  # noqa: D102
        super().set_edge_tensors_(edge_tensors=edge_tensors)

        # calculate edge weights; use self.edge_weights to directly compute on the correct device
        self.edge_weights = {
            side: self.edge_weighting.compute_weights(edge_tensor=edge_tensor)
            for side, edge_tensor in self.edges.items()
        }

    @property
    def device(self) -> torch.device:  # noqa: D102
        return self.node_embeddings.device

    def reset_parameters(self):
        # Reset node embeddings
        self.node_embeddings.reset_parameters()

        for layer in self.layers.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif len(list(layer.parameters())) > 0:
                logging.warning(f'Layer {layer} has parameters, but not reset_parameters() method.')

    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:  # noqa: D102
        # initialize result buffer
        result = {}

        # If no indices are given, do not use indices for any side
        if indices is None:
            indices = {}

        # Perform separately for each side
        for side, x in self.node_embeddings.forward().items():
            # Prepare message passing keyword arguments
            source, target = self.edges[side]
            edge_weights = self.edge_weights[side]
            message_passing_kwargs = {
                'source': source,
                'target': target,
                'edge_weights': edge_weights,
                'num_nodes': self.num_nodes[side],
            }

            # forward pass through all layers
            if side in self.layers.keys():
                layers = self.layers[side] if side in self.layers.keys() else []
            else:
                logging.warning(f'No layers for side {side}')
                layers = []
            for layer in layers:
                if isinstance(layer, MessagePassingBlock):
                    x = layer.forward(x, **message_passing_kwargs)
                else:
                    x = layer.forward(x)

            # Select indices if requested
            this_indices = indices.get(side)
            if this_indices is not None:
                x = x[this_indices]

            # link in result
            result[side] = x
        return result
