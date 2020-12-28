# coding=utf-8
import unittest
from typing import Optional
from unittest import SkipTest

import torch

from base import B, GenericTest, TestTests
from kgm.data import get_erdos_renyi
from kgm.modules.graph import EdgeWeighting, GCNBlock, IdentityMessageCreator, InverseSourceOutDegreeWeighting, InverseTargetInDegreeWeighting, LinearMessageCreator, MessageAggregator, MessageCreator, MessagePassingBlock, MissingEdgeTypesException, NodeUpdater, OnlyUpdate, SumAggregator, SymmetricWeighting


class _CommonGraphTests(GenericTest[B]):
    """Common unittests for graph-based blocks."""
    #: The number of input features
    in_channels: int = 7

    #: The number of message features
    message_channels: int = 13

    #: The number of output features
    out_channels: int = 5

    #: The number of relations
    num_relations: int = 3

    #: The number of entities
    num_entities: int = 17

    #: The edge tensor
    edge_tensor: torch.LongTensor

    #: The edge type
    edge_type: torch.LongTensor

    #: The edge weights
    edge_weights: torch.FloatTensor

    #: The node features
    x: torch.FloatTensor

    def setUp(self):
        super().setUp()
        graph = get_erdos_renyi(num_entities=self.num_entities, num_relations=self.num_relations, p=.4)
        self.edge_tensor = torch.stack([graph.triples[:, i] for i in [0, 2]])
        self.edge_type = graph.triples[:, 1]
        self.edge_weights = torch.rand(self.edge_tensor.shape[1])
        assert self.edge_tensor.shape == (2, graph.triples.shape[0])
        assert self.edge_weights.shape == (self.edge_tensor.shape[1],)
        assert self.edge_type.shape == (self.edge_tensor.shape[1],)
        self.x = torch.rand(self.num_entities, self.in_channels, requires_grad=True)


class _MessageCreatorTests(_CommonGraphTests[MessageCreator]):
    """Common unittests for MessageCreator subclasses."""

    def _test_create_messages(self, edge_type: Optional[torch.LongTensor]):
        source, target = self.edge_tensor
        try:
            messages = self.instance.create_messages(
                x=self.x,
                source=source,
                target=target,
                edge_type=edge_type,
            )
        except MissingEdgeTypesException as e:
            raise SkipTest(str(e)) from e

        # check shape
        assert messages.shape == (source.shape[0], self.out_channels)

        # test backward
        messages.mean().backward()

    def test_create_messages(self):
        self._test_create_messages(edge_type=None)

    def test_create_messages_type(self):
        self._test_create_messages(edge_type=self.edge_type)


class IdentityMessageCreatorTests(_MessageCreatorTests, unittest.TestCase):
    cls = IdentityMessageCreator

    # Identity message can only handle out_features = in_features
    out_channels = _CommonGraphTests.in_channels


class LinearMessageCreatorTests(_MessageCreatorTests, unittest.TestCase):
    cls = LinearMessageCreator
    kwargs = {
        'in_features': _CommonGraphTests.in_channels,
        'out_features': _CommonGraphTests.out_channels,
    }


class MessageCreatorTestsTests(TestTests, unittest.TestCase):
    base_cls = MessageCreator
    base_test_cls = _MessageCreatorTests


class _MessageAggregatorTests(_CommonGraphTests[MessageAggregator]):
    def _test_aggregate_messages(self, edge_type):
        source, target = self.edge_tensor
        messages = torch.rand(source.shape[0], self.message_channels, requires_grad=True)

        agg = self.instance.aggregate_messages(
            msg=messages,
            source=source,
            target=target,
            edge_type=edge_type,
            num_nodes=self.num_entities,
        )

        # check shape
        assert agg.shape == (self.num_entities, self.message_channels)

        # test backward
        agg.mean().backward()

    def test_aggregate_messages(self):
        self._test_aggregate_messages(edge_type=None)

    def test_aggregate_messages_type(self):
        self._test_aggregate_messages(edge_type=self.edge_type)


class SumAggregatorTests(_MessageAggregatorTests, unittest.TestCase):
    cls = SumAggregator


class MessageAggregatorTestsTests(TestTests, unittest.TestCase):
    base_cls = MessageAggregator
    base_test_cls = _MessageAggregatorTests


class _NodeUpdaterTests(_CommonGraphTests[NodeUpdater]):
    def test_combine(self):
        delta = torch.rand(self.num_entities, self.message_channels, requires_grad=True)
        assert delta.shape[0] == self.x.shape[0]
        x = self.instance.combine(
            x=self.x,
            delta=delta,
        )

        # check shape
        assert x.shape == (self.num_entities, self.out_channels)

        # test backward
        x.mean().backward()


class OnlyUpdateTests(_NodeUpdaterTests, unittest.TestCase):
    cls = OnlyUpdate

    # Only works with out_channels = message_channels
    message_channels = _CommonGraphTests.out_channels


class NodeUpdaterTestsTests(TestTests, unittest.TestCase):
    base_cls = NodeUpdater
    base_test_cls = _NodeUpdaterTests


class _EdgeWeightingTests(_CommonGraphTests[EdgeWeighting]):
    def test_compute_weights(self):
        edge_weights = self.instance.compute_weights(edge_tensor=self.edge_tensor)

        # check shape
        assert edge_weights.shape == (self.edge_tensor.shape[1],)


class InverseTargetInDegreeWeightingTests(_EdgeWeightingTests, unittest.TestCase):
    cls = InverseTargetInDegreeWeighting


class InverseSourceOutDegreeWeightingTests(_EdgeWeightingTests, unittest.TestCase):
    cls = InverseSourceOutDegreeWeighting


class SymmetricWeightingTests(_EdgeWeightingTests, unittest.TestCase):
    cls = SymmetricWeighting


class EdgeWeightingTestsTests(TestTests[EdgeWeighting], unittest.TestCase):
    base_cls = EdgeWeighting
    base_test_cls = _EdgeWeightingTests


class _MessagePassingBlockTests(_CommonGraphTests[MessagePassingBlock]):
    def _test_forward(
        self,
        edge_type: Optional[torch.LongTensor],
        edge_weights: Optional[torch.FloatTensor],
    ):
        source, target = self.edge_tensor
        try:
            y = self.instance.forward(
                x=self.x,
                source=source,
                target=target,
                edge_type=edge_type,
                edge_weights=edge_weights,
            )
        except MissingEdgeTypesException as e:
            raise SkipTest(str(e)) from e

        # check shape
        assert y.shape == (self.num_entities, self.out_channels)

        # test backward
        loss = y.mean()
        loss.backward()

    def test_forward(self):
        self._test_forward(edge_type=None, edge_weights=None)

    def test_forward_type(self):
        self._test_forward(edge_type=self.edge_type, edge_weights=None)

    def test_forward_weights(self):
        self._test_forward(edge_type=None, edge_weights=self.edge_weights)

    def test_forward_type_weights(self):
        self._test_forward(edge_type=self.edge_type, edge_weights=self.edge_weights)


class GCNBlockTests(_MessagePassingBlockTests, unittest.TestCase):
    cls = GCNBlock
    kwargs = {
        'in_features': _CommonGraphTests.in_channels,
        'out_features': _CommonGraphTests.out_channels,
        'use_bias': False,
    }


class GCNBlockBiasTests(_MessagePassingBlockTests, unittest.TestCase):
    cls = GCNBlock
    kwargs = {
        'in_features': _CommonGraphTests.in_channels,
        'out_features': _CommonGraphTests.out_channels,
        'use_bias': True,
    }


class MessagePassingBlockTestsTests(TestTests[MessagePassingBlock], unittest.TestCase):
    base_cls = MessagePassingBlock
    base_test_cls = _MessagePassingBlockTests
