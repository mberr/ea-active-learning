# coding=utf-8
import unittest

import torch
from torch import optim

from kgm.data import get_erdos_renyi
from kgm.utils.torch_utils import csls, filter_edges_by_nodes, get_optimizer_class_by_name, remove_node_from_edges_while_keeping_paths


def test_filter_edges_by_nodes():
    num_nodes = 10
    c = torch.arange(num_nodes).view(-1, 1).repeat(1, num_nodes).view(-1)
    d = torch.arange(num_nodes).view(1, -1).repeat(num_nodes, 1).view(-1)
    edge_tensor = torch.stack([c, d], dim=0)
    negatives = list(range(1, num_nodes, 2))
    filtered_edge_tensor = filter_edges_by_nodes(edge_tensor=edge_tensor, negatives=negatives)

    a = torch.arange(0, num_nodes, 2).view(-1, 1).repeat(1, num_nodes).view(-1)
    b = torch.arange(0, num_nodes, 2).view(1, -1).repeat(num_nodes, 1).view(-1)
    exp_filtered_edge_tensor = torch.stack([a, b], dim=0)

    assert set(zip(*filtered_edge_tensor.tolist())) == set(zip(*exp_filtered_edge_tensor.tolist()))


class RemoveNodeFromEdgesWhileKeepingPathsTests(unittest.TestCase):
    #: The number of entities (i.e. nodes)
    num_entities: int = 128

    #: The number of nodes to remove
    num_to_remove: int = 4

    def setUp(self) -> None:
        self.edge_tensor = get_erdos_renyi(num_entities=self.num_entities, num_relations=1, p=.6).edge_tensor_unique

    def test_remove_node_from_edges_while_keeping_paths(self):
        for i in map(int, torch.randperm(self.num_entities)[:self.num_to_remove]):
            new_edge_tensor = remove_node_from_edges_while_keeping_paths(edge_tensor=self.edge_tensor, node_id=i)

            # check that node has been removed
            assert not (new_edge_tensor == i).any()

    def test_manual_example(self):
        edge_tensor = torch.as_tensor([
            [0, 1],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
        ]).t()
        node_id = 1
        exp_new_edge_tensor = torch.as_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ]).t()
        new_edge_tensor = remove_node_from_edges_while_keeping_paths(edge_tensor=edge_tensor, node_id=node_id)
        assert (new_edge_tensor == exp_new_edge_tensor).all()


class CSLSTests(unittest.TestCase):
    n_left: int = 13
    n_right = int = 17
    k: int = 3

    def setUp(self) -> None:
        self.sim = torch.rand(self.n_left, self.n_right, requires_grad=True)

    def _test_backward(self, sim_n: torch.FloatTensor):
        fake_loss = sim_n.mean()
        fake_loss.backward()

    def test_with_k(self):
        sim_n = csls(sim=self.sim, k=self.k)
        self._test_backward(sim_n=sim_n)

    def test_without_k(self):
        sim_n = csls(sim=self.sim)
        self._test_backward(sim_n=sim_n)


def test_get_optimizer_class_by_name():
    for exp_opt_class in [
        optim.Adadelta,
        optim.Adagrad,
        optim.Adam,
        optim.AdamW,
        optim.SparseAdam,
        optim.Adamax,
        optim.ASGD,
        optim.LBFGS,
        optim.RMSprop,
        optim.Rprop,
        optim.SGD,
    ]:
        name = exp_opt_class.__name__.lower()
        opt_class = get_optimizer_class_by_name(name=name)
        assert opt_class == exp_opt_class
