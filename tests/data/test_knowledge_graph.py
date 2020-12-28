# coding=utf-8
import copy
import unittest

import torch

from kgm.data import EntityAlignment, KnowledgeGraph, KnowledgeGraphAlignmentDataset, exact_self_alignment, get_erdos_renyi
from kgm.data.knowledge_graph import add_inverse_triples, add_self_loops, apply_compaction, compact_columns, compact_graph, compact_knowledge_graph_alignment_dataset, compact_single_alignment, get_synthetic_math_graph, sub_graph_alignment


class CompactionTests(unittest.TestCase):
    def test_apply_compaction(self):
        exp_compact_triples = torch.arange(100).unsqueeze(dim=1)
        triples = exp_compact_triples.clone()
        triples[50:] += 50
        compaction = {old: new for new, old in enumerate(sorted(list(range(50)) + list(range(100, 150))))}
        columns = {0}
        dim = 0
        new_triples = apply_compaction(triples, compaction, columns, dim=dim)
        assert new_triples.shape == exp_compact_triples.shape
        assert (new_triples == exp_compact_triples).all()

    def test_compact_columns(self):
        triples = torch.stack([torch.arange(0, 100, 2) for _ in range(3)], dim=1)
        old_triples = triples.clone()
        assert triples.shape == (50, 3)
        mapping = {f'e_{i // 2}': i for i in range(0, 100, 2)}
        old_mapping = {k: v for k, v in mapping.items()}
        columns = {0, 2}
        new_triples, new_mapping, compaction = compact_columns(triples=triples, label_to_id_mapping=mapping, columns=columns, dim=0)

        # check for side-effects
        assert mapping == old_mapping
        assert (triples == old_triples).all()

        # check new triples
        for c in range(3):
            if c not in columns:
                # Check whether masked columns are unchanged
                assert (new_triples[:, c] == old_triples[:, c]).all()
            else:
                # check if columns have been compacted
                assert set(int(a) for a in new_triples[:, c]) == set(range(50))

        # check if mapping has been updated
        assert new_mapping == {f'e_{i}': i for i in range(50)}

    def test_compact_graph(self):
        triples = torch.stack([torch.arange(0, 100, 2) for _ in range(3)], dim=1)
        old_triples = triples.clone()
        entity_label_to_id = {f'e_{i}': i for i in range(0, 100, 2)}
        old_entity_label_to_id = {k: v for k, v in entity_label_to_id.items()}
        relation_label_to_id = {f'r_{i}': i for i in range(0, 100, 2)}
        old_relation_label_to_id = {k: v for k, v in relation_label_to_id.items()}
        graph = KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
        )
        new_graph, entity_compaction, relation_compaction = compact_graph(graph=graph)

        new_entity_label_to_id = new_graph.entity_label_to_id
        new_relation_label_to_id = new_graph.relation_label_to_id

        # check for no side effects
        assert (triples == old_triples).all()
        assert old_entity_label_to_id == entity_label_to_id
        assert old_relation_label_to_id == relation_label_to_id

        # check entity compaction
        assert set(entity_compaction.keys()) == set(old_entity_label_to_id.values())
        assert set(entity_compaction.values()) == set(new_entity_label_to_id.values())

        # check relation compaction
        assert set(relation_compaction.keys()) == set(old_relation_label_to_id.values())
        assert set(relation_compaction.values()) == set(new_relation_label_to_id.values())

        # check compacted entity_label_to id
        assert len(new_entity_label_to_id) == len(old_entity_label_to_id)
        assert set(new_entity_label_to_id.values()) == set(range(50))
        assert set(new_entity_label_to_id.keys()) == set(old_entity_label_to_id.keys())

        # check compacted entity_label_to id
        assert len(new_relation_label_to_id) == len(old_relation_label_to_id)
        assert set(new_relation_label_to_id.values()) == set(range(50))
        assert set(new_relation_label_to_id.keys()) == set(old_relation_label_to_id.keys())

        exp_triples = torch.stack([torch.arange(50) for _ in range(3)], dim=1)
        assert (new_graph.triples == exp_triples).all()

    def test_compact_single_alignment(self):
        single_alignment = torch.stack([torch.arange(0, i * 50, i) for i in (1, 2)], dim=0)
        match_compaction = None
        ref_compaction = {2 * i: i for i in range(50)}
        assert single_alignment.shape == (2, 50)
        new_alignment = compact_single_alignment(
            single_alignment=single_alignment,
            left_compaction=match_compaction,
            right_compaction=ref_compaction,
        )
        exp_new_alignment = torch.stack([torch.arange(50) for _ in range(2)], dim=0)
        assert (new_alignment == exp_new_alignment).all()

    def test_compact(self):
        num_match_triples = 50
        match_triples = torch.stack([torch.arange(i, 6 * num_match_triples, 6) for i in (0, 2, 4)], dim=1)
        match_entity_label_to_id = {f'me_{2 * i}': 6 * i + 0 for i in range(num_match_triples)}
        match_entity_label_to_id.update({f'me_{2 * i + 1}': 6 * i + 4 for i in range(num_match_triples)})
        match_relation_label_to_id = {f'mr_{i}': 6 * i + 2 for i in range(num_match_triples)}
        assert len(set(match_entity_label_to_id.values())) == len(match_entity_label_to_id.values())
        assert len(set(match_relation_label_to_id.values())) == len(match_relation_label_to_id.values())
        match_graph = KnowledgeGraph(
            triples=match_triples,
            entity_label_to_id=match_entity_label_to_id,
            relation_label_to_id=match_relation_label_to_id,
        )
        assert match_graph.triples is not None
        assert match_graph.entity_label_to_id is not None
        assert match_graph.relation_label_to_id is not None

        num_ref_triples = 60
        ref_triples = torch.stack([torch.arange(i, 6 * num_ref_triples, 6) for i in (1, 3, 5)], dim=1)
        ref_entity_label_to_id = {f're_{2 * i}': 6 * i + 1 for i in range(num_ref_triples)}
        ref_entity_label_to_id.update({f're_{2 * i + 1}': 6 * i + 5 for i in range(num_ref_triples)})
        ref_relation_label_to_id = {f'rr_{i}': 6 * i + 3 for i in range(num_ref_triples)}
        assert len(set(ref_entity_label_to_id.values())) == len(ref_entity_label_to_id.values())
        assert len(set(ref_relation_label_to_id.values())) == len(ref_relation_label_to_id.values())
        ref_graph = KnowledgeGraph(
            triples=ref_triples,
            entity_label_to_id=ref_entity_label_to_id,
            relation_label_to_id=ref_relation_label_to_id,
        )
        assert ref_graph.triples is not None
        assert ref_graph.entity_label_to_id is not None
        assert ref_graph.relation_label_to_id is not None

        entity_alignment_train = torch.stack([torch.arange(0, 60, 6), torch.arange(1, 60, 6)], dim=0)
        entity_alignment_test = torch.stack([torch.arange(4, 30, 6), torch.arange(5, 30, 6)], dim=0)
        assert entity_alignment_train.shape == (2, 10)
        assert entity_alignment_test.shape == (2, 5)
        assert set(map(int, entity_alignment_train[0])).issubset(set(match_entity_label_to_id.values()))
        assert set(map(int, entity_alignment_train[1])).issubset(set(ref_entity_label_to_id.values()))
        assert set(map(int, entity_alignment_test[0])).issubset(set(match_entity_label_to_id.values()))
        assert set(map(int, entity_alignment_test[1])).issubset(set(ref_entity_label_to_id.values()))
        alignment = EntityAlignment(
            train=entity_alignment_train,
            test=entity_alignment_test,
        )
        assert alignment.train is not None
        assert alignment.test is not None

        new_match_graph, new_ref_graph, new_alignment = compact_knowledge_graph_alignment_dataset(left_graph=match_graph, right_graph=ref_graph, alignment=alignment)

        # Check match graph
        # Triples
        exp_new_match_graph_triples = torch.stack([
            torch.arange(0, 2 * num_match_triples, 2),
            torch.arange(num_match_triples),
            torch.arange(1, 2 * num_match_triples, 2),
        ], dim=1)
        assert set(new_match_graph.entity_label_to_id.values()) == set(range(len(match_entity_label_to_id)))
        assert set(new_match_graph.relation_label_to_id.values()) == set(range(len(match_relation_label_to_id)))
        assert (new_match_graph.triples == exp_new_match_graph_triples).all()

        # label to id
        exp_match_entity_label_to_id = {f'me_{i}': i for i in range(2 * num_match_triples)}
        exp_match_relation_label_to_id = {f'mr_{i}': i for i in range(num_match_triples)}
        assert new_match_graph.entity_label_to_id == exp_match_entity_label_to_id
        assert new_match_graph.relation_label_to_id == exp_match_relation_label_to_id

        # Check ref graph
        exp_new_ref_graph_triples = torch.stack([
            torch.arange(0, 2 * num_ref_triples, 2),
            torch.arange(num_ref_triples),
            torch.arange(1, 2 * num_ref_triples, 2),
        ], dim=1)
        assert set(new_ref_graph.entity_label_to_id.values()) == set(range(len(ref_entity_label_to_id)))
        assert set(new_ref_graph.relation_label_to_id.values()) == set(range(len(ref_relation_label_to_id)))
        assert (new_ref_graph.triples == exp_new_ref_graph_triples).all()

        # label to id
        exp_ref_entity_label_to_id = {f're_{i}': i for i in range(2 * num_ref_triples)}
        exp_ref_relation_label_to_id = {f'rr_{i}': i for i in range(num_ref_triples)}
        assert new_ref_graph.entity_label_to_id == exp_ref_entity_label_to_id
        assert new_ref_graph.relation_label_to_id == exp_ref_relation_label_to_id

        # check alignment
        new_entity_alignment_train = new_alignment.train
        new_entity_alignment_test = new_alignment.test
        exp_entity_alignment_train = torch.stack([torch.arange(0, 20, 2), torch.arange(0, 20, 2)], dim=0)
        exp_entity_alignment_test = torch.stack([torch.arange(1, 10, 2), torch.arange(1, 10, 2)], dim=0)
        assert (new_entity_alignment_train == exp_entity_alignment_train).all()
        assert (new_entity_alignment_test == exp_entity_alignment_test).all()


def test_exact_self_alignment():
    num_entities = 16
    num_relations = 8
    graph = get_erdos_renyi(num_entities=num_entities, num_relations=num_relations, p=.1)
    kga = exact_self_alignment(graph=graph)
    left = kga.left_graph
    right = kga.right_graph
    for g in (left, right):
        assert g.num_entities == num_entities
        assert g.num_relations == num_relations
    assert left.num_triples == right.num_triples


def test_add_self_loops():
    self_loop_relation_name = 'special_name'
    num_entities = 16
    num_relations = 8
    graph = get_erdos_renyi(num_entities=num_entities, num_relations=num_relations, p=.1)
    triples = graph.triples.detach().clone()
    relation_label_to_id = copy.deepcopy(graph.relation_label_to_id)
    new_triples, new_relation_label_to_id = add_self_loops(triples=triples, entity_label_to_id=graph.entity_label_to_id, relation_label_to_id=relation_label_to_id, self_loop_relation_name=self_loop_relation_name)

    # check relation_label_to_id
    # not in-place
    assert id(new_relation_label_to_id) != id(relation_label_to_id)

    # correct keys
    assert set(new_relation_label_to_id.keys()) == set(relation_label_to_id.keys()).union({self_loop_relation_name})

    # correct values
    assert set(new_relation_label_to_id.values()) == set(relation_label_to_id.values()).union({max(relation_label_to_id.values()) + 1})

    # check new_triples
    assert new_triples.shape[1] == graph.triples.shape[1]
    assert new_triples.shape[0] == graph.triples.shape[0] + num_entities

    assert (new_triples[:graph.triples.shape[0]] == graph.triples).all()


def test_add_inverse_triples():
    num_entities = 16
    num_relations = 8
    inverse_relation_postfix = '_inv'
    graph = get_erdos_renyi(num_entities=num_entities, num_relations=num_relations, p=.1)
    triples = graph.triples.detach().clone()
    relation_label_to_id = copy.deepcopy(graph.relation_label_to_id)
    new_triples, new_relation_label_to_id = add_inverse_triples(triples=triples, relation_label_to_id=relation_label_to_id, inverse_relation_postfix=inverse_relation_postfix)

    # check relation_label_to_id
    # not in-place
    assert id(new_relation_label_to_id) != id(relation_label_to_id)

    # correct keys
    assert set(new_relation_label_to_id.keys()) == set(relation_label_to_id.keys()).union(map(lambda s: s + inverse_relation_postfix, relation_label_to_id.keys()))

    # correct values
    assert set(new_relation_label_to_id.values()) == set(relation_label_to_id.values()).union(map(lambda _id: _id + graph.num_relations, relation_label_to_id.values()))

    # check new_triples
    assert new_triples.shape[1] == graph.triples.shape[1]
    assert new_triples.shape[0] == 2 * graph.triples.shape[0]

    assert (new_triples[:graph.triples.shape[0]] == graph.triples).all()


class _KnowledgeGraphGenerationTests:
    num_entities: int = 7

    def setUp(self) -> None:
        self.kg = self.create()

    def create(self) -> KnowledgeGraph:
        raise NotImplementedError

    def test_triples(self):
        assert self.kg.triples.shape == (self.kg.num_triples, 3)

    def test_entity_label_to_id(self):
        assert len(self.kg.entity_label_to_id) == self.num_entities
        for h, r, t in self.kg.triples.tolist():
            assert h in self.kg.entity_label_to_id.values()
            assert t in self.kg.entity_label_to_id.values()

    def test_num_entities(self):
        assert self.num_entities == self.kg.num_entities


class ErdosRenyiTests(_KnowledgeGraphGenerationTests, unittest.TestCase):
    num_relations: int = 3

    def create(self) -> KnowledgeGraph:
        return get_erdos_renyi(num_entities=self.num_entities, num_relations=self.num_relations)

    def test_relations(self):
        assert self.kg.num_relations == self.num_relations


class MathTests(_KnowledgeGraphGenerationTests, unittest.TestCase):
    def create(self) -> KnowledgeGraph:
        return get_synthetic_math_graph(num_entities=self.num_entities)


class _AlignmentGenerationTests:
    num_entities: int = 33
    num_relations: int = 3

    def setUp(self) -> None:
        self.base = get_erdos_renyi(num_entities=self.num_entities, num_relations=self.num_relations, p=.3)
        self.dataset = self.create()

    def create(self) -> KnowledgeGraphAlignmentDataset:
        raise NotImplementedError

    def test_alignment(self):
        assert self.dataset.alignment.train.shape == (2, self.dataset.alignment.num_train)
        assert self.dataset.alignment.test.shape == (2, self.dataset.alignment.num_test)

    def test_graphs(self):
        for side, graph in self.dataset.graphs.items():
            assert graph.triples.shape == (graph.num_triples, 3)


class ExactSelfAlignmentTests(_AlignmentGenerationTests, unittest.TestCase):
    def create(self) -> KnowledgeGraphAlignmentDataset:
        return exact_self_alignment(graph=self.base, train_percentage=0.5)


class SubGraphAlignmentTests(_AlignmentGenerationTests, unittest.TestCase):
    def create(self) -> KnowledgeGraphAlignmentDataset:
        return sub_graph_alignment(graph=self.base)
