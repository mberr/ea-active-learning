import logging
from typing import Optional, Tuple, Type, Union

import torch
from torch import optim

from kgm.utils.common import get_subclass_by_name


def filter_edges_by_nodes(
    edge_tensor: torch.LongTensor,
    negatives: Union[Tuple[int, ...], torch.LongTensor],
) -> torch.LongTensor:
    """Filter out edges containing any of the listed nodes."""
    if not torch.is_tensor(negatives):
        negatives = torch.as_tensor(data=negatives, dtype=torch.long, device=edge_tensor.device)
    left_mask = ~(edge_tensor.view(2, -1, 1) == negatives.view(1, 1, -1)).any(dim=2).any(dim=0)
    edge_tensor = edge_tensor[:, left_mask]
    return edge_tensor


def remove_node_from_edges_while_keeping_paths(
    edge_tensor: torch.LongTensor,
    node_id: int,
) -> torch.LongTensor:
    """
    Removes a node from a graph, and adds edges between all nodes incident to it.

    More precisely, for e being the node to remove, the following datalog expression defines the edges which are added

        (s, t) <- (s, e), (e, t)

    :param edge_tensor: shape: (2, num_edges)
        The edge tensor.
    :param node_id:
        The node id to remove.

    :return new_edge_tensor: shape (2, num_edges_new)
        The new edge tensor.
    """
    source, target = edge_tensor

    # check which edges are incident to e
    source_e = (source == node_id)
    target_e = (target == node_id)

    # keep edges without e
    edge_tensor = edge_tensor[:, ~(source_e | target_e)]

    # if (s, e) and (e, t) add (s, t)
    source_neighbours = source[target_e].unique()
    target_neighbours = target[source_e].unique()
    n_sources, n_targets = source_neighbours.shape[0], target_neighbours.shape[0]
    new_edges = torch.stack([
        source_neighbours.view(-1, 1).repeat(1, n_targets),
        target_neighbours.view(1, -1).repeat(n_sources, 1),
    ], dim=0)

    # filter self-edges
    new_edges = new_edges[:, new_edges[0, :] != new_edges[1, :]]

    edge_tensor = torch.cat([edge_tensor, new_edges], dim=1)

    # remove duplicate edges
    edge_tensor = torch.unique(edge_tensor, dim=1)

    return edge_tensor


def get_device(
    device: Union[None, str, torch.device],
) -> torch.device:
    """Resolve the device, either specified as name, or device."""
    if device is None:
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device=device)
    assert isinstance(device, torch.device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        logging.warning(f'Requested device {device}, but CUDA is unavailable. Falling back to cpu.')
        device = torch.device('cpu')
    return device


def softmax_entropy_from_logits(
    logits: torch.FloatTensor,
    dim=1,
    temperature: Optional[float] = None,
) -> torch.FloatTensor:
    if temperature is not None:
        logits = logits / temperature
    p = logits.softmax(dim=dim)
    log_p = logits.log_softmax(dim=dim)
    return -(p * log_p).sum(dim=dim)


def csls(
    sim: torch.FloatTensor,
    k: Optional[int] = 1,
) -> torch.FloatTensor:
    """
    Apply CSLS normalization to a similarity matrix.

    .. math::
        csls[i, j] = 2*sim[i, j] - avg(top_k(sim[i, :])) - avg(top_k(sim[:, j]))

    :param sim: shape: (d1, ..., dk)
        Similarity matrix.
    :param k:
        The number of top-k elements to use for correction.

    :return:
        The normalized similarity matrix.
    """
    if k is None:
        return sim

    # Empty similarity matrix
    if sim.numel() < 1:
        return sim

    # compensate for subtraction
    sim = sim.ndimension() * sim

    # Subtract average over top-k similarities for each mode of the tensors.
    old_sim = sim
    for dim, size in enumerate(sim.size()):
        sim = sim - old_sim.topk(k=min(k, size), dim=dim, largest=True, sorted=False).values.mean(dim=dim, keepdim=True)

    return sim


def resolve_device_from_to_kwargs(args, kwargs):
    """Wrapper of to(*args, **kwargs) parameter resolution."""
    return torch._C._nn._parse_to(*args, **kwargs)[0]


def _guess_num_nodes(
    num_nodes: Optional[int],
    source: Optional[torch.LongTensor] = None,
    target: Optional[torch.LongTensor] = None,
) -> int:
    if num_nodes is not None:
        return num_nodes
    if source is None and target is None:
        raise ValueError('If no num_nodes are given, either source, or target must be given!')
    return max(x.max().item() for x in (source, target) if x is not None)


def get_optimizer_class_by_name(name: str) -> Type[optim.Optimizer]:
    return get_subclass_by_name(base_class=optim.Optimizer, name=name, normalizer=str.lower)


def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1) -> torch.Tensor:
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
