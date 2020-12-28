# coding=utf-8
import logging
from typing import Optional

import torch
from torch import nn

__all__ = [
    'EdgeWeighting',
    'InverseSourceOutDegreeWeighting',
    'InverseTargetInDegreeWeighting',
    'SymmetricWeighting',
]

_LOGGER = logging.getLogger(name=__name__)


class EdgeWeighting:
    """A scheme to compute edge weights given an edge tensor."""

    @staticmethod
    def compute_weights(edge_tensor: torch.LongTensor) -> torch.FloatTensor:
        """Compute edge weights from the edge tensor.

        :param edge_tensor: shape: (2, num_edges)
            The edge tensor.

        :return: shape: (num_edges,)
            The edge weights.
        """
        raise NotImplementedError


class InverseTargetInDegreeWeighting(EdgeWeighting):
    """Weights edges by the inverse in-degree of the target node.

    Thereby, the weights of incoming messages sum up to one.
    """

    @staticmethod
    def compute_weights(edge_tensor: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # compute in-degree of the target node
        edge_id_to_target_node_id, target_in_degree = torch.unique(edge_tensor[1], sorted=False, return_counts=True, return_inverse=True)[1:]

        return target_in_degree.float().reciprocal()[edge_id_to_target_node_id]


class InverseSourceOutDegreeWeighting(EdgeWeighting):
    """Weights edges by the inverse out-degree of the source node.

    Thereby, the weights of outgoing messages sum up to one.
    """

    @staticmethod
    def compute_weights(edge_tensor: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # compute out-degree of the source node
        edge_id_to_source_node_id, source_out_degree = torch.unique(edge_tensor[0], sorted=False, return_counts=True, return_inverse=True)[1:]

        return source_out_degree.float().reciprocal()[edge_id_to_source_node_id]


class SymmetricWeighting(EdgeWeighting):
    """Weights edges by the product of the inverse square root of degrees on target and source node."""

    @staticmethod
    def compute_weights(edge_tensor: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # product of inverse square roots
        return InverseTargetInDegreeWeighting.compute_weights(edge_tensor=edge_tensor).sqrt() * InverseSourceOutDegreeWeighting.compute_weights(edge_tensor=edge_tensor).sqrt()


class MissingEdgeTypesException(BaseException):
    """Class requires edge information."""

    def __init__(self, cls):
        super().__init__(f'{cls.__name__} requires passing edge types.')


class MessageCreator(nn.Module):
    def reset_parameters(self) -> None:
        pass

    def create_messages(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Create messages.

        :param x: shape: (num_nodes, node_embedding_dim)
            The node representations.

        :param source: (num_edges,)
            The source indices for each edge.

        :param target: shape: (num_edges,)
            The target indices for each edge.

        :param edge_type: shape: (num_edges,)
            The edge type for each edge.

        :return: shape: (num_edges, message_dim)
            The messages source -> target.
        """
        raise NotImplementedError


class IdentityMessageCreator(MessageCreator):
    def create_messages(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        return x.index_select(dim=0, index=source)


class LinearMessageCreator(MessageCreator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)

    def create_messages(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        x = self.linear.forward(x)
        return x.index_select(dim=0, index=source)


def _guess_num_nodes(num_nodes: Optional[int], source: torch.LongTensor, target: torch.LongTensor) -> int:
    if num_nodes is not None:
        return num_nodes
    return max(source.max().item(), target.max().item()) + 1


class MessageAggregator(nn.Module):
    def reset_parameters(self) -> None:
        pass

    def aggregate_messages(
        self,
        msg: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: Optional[torch.LongTensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Aggregate messages per node.

        :param msg: shape: (num_edges, message_dim)
            The messages source -> target.

        :param source: (num_edges,)
            The source indices for each edge.

        :param target: shape: (num_edges,)
            The target indices for each edge.

        :param edge_type: shape: (num_edges,)
            The edge type for each edge.

        :param num_nodes: >0
            The number of nodes. If None is provided tries to guess the number of nodes by max(source.max(), target.max()) + 1

        :return: shape: (num_nodes, update_dim)
            The node updates.
        """
        raise NotImplementedError


class SumAggregator(MessageAggregator):
    def aggregate_messages(
        self,
        msg: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: Optional[torch.LongTensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        num_nodes = _guess_num_nodes(num_nodes=num_nodes, source=source, target=target)
        dim = msg.shape[1]
        return torch.zeros(num_nodes, dim, dtype=msg.dtype, device=msg.device).index_add_(dim=0, index=target, source=msg)


class NodeUpdater(nn.Module):
    def reset_parameters(self) -> None:
        pass

    def combine(
        self,
        x: torch.FloatTensor,
        delta: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Update node representations.

        :param x: shape: (num_nodes, node_embedding_dim)
            The node representations.

        :param delta: (num_nodes, update_dim)
            The node updates.

        :return: shape: (num_nodes, new_node_embedding_dim)
            The new node representations.
        """
        raise NotImplementedError


class OnlyUpdate(NodeUpdater):
    def combine(
        self,
        x: torch.FloatTensor,
        delta: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return delta


class MessagePassingBlock(nn.Module):
    def __init__(
        self,
        message_creator: MessageCreator,
        message_aggregator: MessageAggregator,
        node_updater: NodeUpdater,
    ):
        super().__init__()

        # Bind sub-modules
        self.message_creator = message_creator
        self.message_aggregator = message_aggregator
        self.node_updater = node_updater

    def reset_parameters(self) -> None:
        self.message_creator.reset_parameters()
        self.message_aggregator.reset_parameters()
        self.node_updater.reset_parameters()

    def forward(
        self,
        x: torch.FloatTensor,
        source: torch.LongTensor,
        target: torch.LongTensor,
        edge_type: Optional[torch.LongTensor] = None,
        edge_weights: Optional[torch.FloatTensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.FloatTensor:
        """
        Calculate new node representations by message passing.

       :param x: shape: (num_nodes, node_embedding_dim)
            The node representations.

        :param source: (num_edges,)
            The source indices for each edge.

        :param target: shape: (num_edges,)
            The target indices for each edge.

        :param edge_type: shape: (num_edges,)
            The edge type for each edge.

        :param edge_weights: shape (num_edges,)
            The edge weights.

        :return: shape: (num_nodes, new_node_embedding_dim)
            The new node representations.
        """
        # create messages
        messages = self.message_creator.create_messages(x=x, source=source, target=target, edge_type=edge_type)

        # apply edge weights
        if edge_weights is not None:
            messages = messages * edge_weights.unsqueeze(dim=-1)

        # aggregate
        delta = self.message_aggregator.aggregate_messages(msg=messages, source=source, target=target, edge_type=edge_type, num_nodes=num_nodes)
        del messages

        return self.node_updater.combine(x=x, delta=delta)


class GCNBlock(MessagePassingBlock):
    """
    GCN model roughly following https://arxiv.org/abs/1609.02907.

    Notice that this module does only the message passing part, and does **not** apply a non-linearity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool,
    ):
        super().__init__(
            message_creator=LinearMessageCreator(
                in_features=in_features,
                out_features=out_features,
                use_bias=use_bias
            ),
            message_aggregator=SumAggregator(),
            node_updater=OnlyUpdate()
        )
