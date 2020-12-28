# coding=utf-8
"""
Loading utilities for knowledge graphs.
"""
import abc
import enum
import logging
import os
import pathlib
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from os import path
from typing import Collection, List, Mapping, Optional, Set, TextIO, Tuple, Union

import networkx
import numpy
import requests
import torch
from networkx import nx

from ..utils.common import get_all_subclasses, get_subclass_by_name


@enum.unique
class MatchSideEnum(str, enum.Enum):
    #: The left side
    left = 'left'

    #: The right side
    right = 'right'


SIDES = (MatchSideEnum.left, MatchSideEnum.right)


def get_other_side(side: MatchSideEnum) -> MatchSideEnum:
    """Get the enum of the other side."""
    return MatchSideEnum.left if side == MatchSideEnum.right else MatchSideEnum.right


class Extractor:
    def extract(self, archive_path: str, cache_root: Optional[str] = None, force_extraction: bool = False):
        if cache_root is None:
            cache_root = path.dirname(archive_path)
        flag_file = path.join(cache_root, '.finished_extraction')
        if not path.isfile(flag_file) or force_extraction:
            self._extract(archive_path=archive_path, cache_root=cache_root)
            logging.info(f'Extracted {archive_path} to {cache_root}.')

            # create extraction flag file
            pathlib.Path(flag_file).touch()
        else:
            logging.info(f'Skipping extraction due to existing file {flag_file}.')

    def _extract(self, archive_path: str, cache_root: str):
        raise NotImplementedError


class ZipExtractor(Extractor):
    def _extract(self, archive_path: str, cache_root: str):
        with zipfile.ZipFile(file=archive_path) as zf:
            zf.extractall(path=cache_root)


class TarExtractor(Extractor):
    def _extract(self, archive_path: str, cache_root: str):
        with open(archive_path, 'rb') as archive_file:
            with tarfile.open(fileobj=archive_file) as tf:
                tf.extractall(path=cache_root)


def add_self_loops(
    triples: torch.LongTensor,
    entity_label_to_id: Mapping[str, int],
    relation_label_to_id: Mapping[str, int],
    self_loop_relation_name: Optional[str] = None,
) -> Tuple[torch.LongTensor, Mapping[str, int]]:
    """Add self loops with dummy relation.

    For each entity e, add (e, self_loop, e).

    :param triples: shape: (n, 3)
         The triples.
    :param entity_label_to_id:
        The mapping from entity labels to ids.
    :param relation_label_to_id:
        The mapping from relation labels to ids.
    :param self_loop_relation_name:
        The name of the self-loop relation. Must not exist.

    :return:
        cat(triples, self_loop_triples)
        updated mapping
    """
    if self_loop_relation_name is None:
        self_loop_relation_name = 'self_loop'
    _, p, _ = triples[:, 0], triples[:, 1], triples[:, 2]

    # check if name clashes might occur
    if self_loop_relation_name in relation_label_to_id.keys():
        raise AssertionError(f'There exists a relation "{self_loop_relation_name}".')

    # Append inverse relations to translation table
    max_relation_id = max(relation_label_to_id.values())
    updated_relation_label_to_id = {r_label: r_id for r_label, r_id in relation_label_to_id.items()}
    self_loop_relation_id = max_relation_id + 1
    updated_relation_label_to_id.update({self_loop_relation_name: self_loop_relation_id})
    assert len(updated_relation_label_to_id) == len(relation_label_to_id) + 1

    # create self-loops triples
    assert (p <= max_relation_id).all()
    e = torch.tensor(sorted(entity_label_to_id.values()), dtype=torch.long)
    p_self_loop = torch.ones_like(e) * self_loop_relation_id
    self_loop_triples = torch.stack([e, p_self_loop, e], dim=1)

    all_triples: torch.LongTensor = torch.cat([triples, self_loop_triples], dim=0)

    return all_triples, updated_relation_label_to_id


def add_inverse_triples(
    triples: torch.LongTensor,
    relation_label_to_id: Mapping[str, int],
    inverse_relation_postfix: Optional[str] = None,
) -> Tuple[torch.LongTensor, Mapping[str, int]]:
    """Create and append inverse triples.

    For each triple (s, p, o), an inverse triple (o, p_inv, s) is added.

    :param triples: shape: (n, 3)
        The triples.
    :param relation_label_to_id:
        The mapping from relation labels to ids.
    :param inverse_relation_postfix:
        A postfix to use for creating labels for the inverse relations.

    :return: cat(triples, inverse_triples)
    """
    if inverse_relation_postfix is None:
        inverse_relation_postfix = '_inv'
    assert len(inverse_relation_postfix) > 0

    s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]

    # check if name clashes might occur
    suspicious_relations = sorted(k for k in relation_label_to_id.keys() if k.endswith('_inv'))
    if len(suspicious_relations) > 0:
        raise AssertionError(
            f'Some of the inverse relations did already exist! Suspicious relations: {suspicious_relations}')

    # Append inverse relations to translation table
    num_relations = len(relation_label_to_id)
    updated_relation_label_to_id = {r_label: r_id for r_label, r_id in relation_label_to_id.items()}
    updated_relation_label_to_id.update({r_label + inverse_relation_postfix: r_id + num_relations for r_label, r_id in relation_label_to_id.items()})
    assert len(updated_relation_label_to_id) == 2 * num_relations

    # create inverse triples
    assert (p < num_relations).all()
    p_inv = p + num_relations
    inverse_triples = torch.stack([o, p_inv, s], dim=1)

    all_triples: torch.LongTensor = torch.cat([triples, inverse_triples], dim=0)

    return all_triples, updated_relation_label_to_id


@dataclass
class KnowledgeGraph:
    """A knowledge graph."""
    #: The triples, shape: (n, 3)
    triples: torch.LongTensor

    #: The mapping from entity labels to IDs
    entity_label_to_id: Optional[Mapping[str, int]]

    #: The mapping from relations labels to IDs
    relation_label_to_id: Optional[Mapping[str, int]]

    #: Language code of the knowledge graph (e.g. zh, en, ...)
    lang_code: Optional[str] = None

    #: Dataset name
    dataset_name: Optional[str] = None

    #: Dataset subset name
    subset_name: Optional[str] = None

    #: Whether inverse triples have been added
    inverse_triples: bool = False

    #: Whether self-loops have been added.
    self_loops: bool = False

    _unique_edge_tensor: torch.LongTensor = None

    _edge_tensor_weights: torch.LongTensor = None

    @property
    def num_triples(self) -> int:
        return self.triples.shape[0]

    @property
    def num_entities(self) -> int:
        return len(self.entity_label_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_label_to_id)

    @property
    def edge_tensor_unique(self) -> torch.LongTensor:
        """
        :return: shape: (2. num_edges)
            The tensor of unique edges, (source, target) pairs.
        """
        if self._unique_edge_tensor is None:
            self.extract_edge_info()
        return self._unique_edge_tensor

    @property
    def edge_weights(self) -> torch.LongTensor:
        if self._edge_tensor_weights is None:
            self.extract_edge_info()
        return self._edge_tensor_weights

    def extract_edge_info(self):
        self._unique_edge_tensor, self._edge_tensor_weights = torch.stack([self.triples[:, i] for i in (0, 2)], dim=0).unique(dim=1, return_counts=True)
        self._edge_tensor_weights = self._edge_tensor_weights.float()

    def with_inverse_triples(
        self,
        inverse_relation_postfix: Optional[str] = None,
    ) -> 'KnowledgeGraph':
        """Return a KG with added inverse triples, if not already contained. Otherwise return reference to self."""
        assert not self.self_loops
        if self.inverse_triples:
            return self
        else:
            enriched_triples, enriched_relation_label_to_id = add_inverse_triples(
                triples=self.triples,
                relation_label_to_id=self.relation_label_to_id,
                inverse_relation_postfix=inverse_relation_postfix,
            )
            return KnowledgeGraph(
                triples=enriched_triples,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=enriched_relation_label_to_id,
                inverse_triples=True,
                self_loops=False,
                lang_code=self.lang_code,
                dataset_name=self.dataset_name,
                subset_name=self.subset_name
            )

    def with_self_loops(
        self,
        self_loop_relation_name: Optional[str] = None,
    ) -> 'KnowledgeGraph':
        """Return a KG with added self-loops, if not already contained. Otherwise return reference to self."""
        if self.self_loops:
            return self
        else:
            enriched_triples, enriched_relation_label_to_id = add_self_loops(
                triples=self.triples,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=self.relation_label_to_id,
                self_loop_relation_name=self_loop_relation_name,
            )
            return KnowledgeGraph(
                triples=enriched_triples,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=enriched_relation_label_to_id,
                inverse_triples=self.inverse_triples,
                self_loops=True,
                lang_code=self.lang_code,
                dataset_name=self.dataset_name,
                subset_name=self.subset_name
            )

    def to_networkx(self) -> networkx.DiGraph:
        """Converts the graph to networkx graph."""
        g = networkx.DiGraph()
        g.add_nodes_from(self.entity_label_to_id.values())
        edges = self.edge_tensor_unique.detach().cpu().t().tolist()
        weights = self.edge_weights.detach().cpu().tolist()
        weighted_edges = map(lambda e: (e[0][0], e[0][1], e[1]), zip(edges, weights))
        g.add_weighted_edges_from(weighted_edges)
        return g

    def __str__(self):
        return f'{self.__class__.__name__}(num_triples={self.num_triples}, num_entities={self.num_entities}, num_relations={self.num_relations}, inverse_triples={self.inverse_triples}, self_loops={self.self_loops})'


@dataclass
class EntityAlignment:
    """An entity alignment between two knowledge graphs."""
    #: The entity alignment used for training, shape: (2, num_train_alignments)
    train: torch.LongTensor

    #: The entity alignment used for testing, shape: (2, num_test_alignments)
    test: torch.LongTensor

    #: The entity alignment used for validation, shape: (2, num_validation_alignments)
    _validation: Optional[torch.LongTensor] = None

    @property
    def validation(self) -> torch.LongTensor:
        if self._validation is None:
            return torch.empty(2, 0, dtype=torch.long, device=self.train.device)
        return self._validation

    @property
    def num_train(self) -> int:
        return self.train.shape[1]

    @property
    def num_validation(self) -> int:
        return self.validation.shape[1]

    @property
    def num_test(self) -> int:
        return self.test.shape[1]

    @property
    def all(self) -> torch.LongTensor:
        return torch.cat([self.train, self.validation, self.test], dim=1)

    def to_dict(self) -> Mapping[str, torch.LongTensor]:
        result = {
            'train': self.train,
            'test': self.test,
        }
        if self._validation is not None:
            result['validation'] = self.validation
        return result

    def validation_split(self, train_ratio: float, seed: Optional[int] = None) -> 'EntityAlignment':
        if train_ratio <= 0. or train_ratio >= 1.:
            raise ValueError(f'ratio must be in (0, 1), but is {train_ratio}')
        return validation_split(alignment=self, train_ratio=train_ratio, seed=seed)

    def __str__(self):
        return f'{self.__class__.__name__}(num_train={self.num_train}, num_test={self.num_test}, num_val={self.num_validation})'


def apply_compaction(
    triples: Optional[torch.LongTensor],
    compaction: Mapping[int, int],
    columns: Union[int, Collection[int]],
    dim: int = 0,
) -> Optional[torch.LongTensor]:
    if compaction is None or triples is None:
        return triples
    if isinstance(columns, int):
        columns = [columns]
    if dim not in {0, 1}:
        raise KeyError(dim)
    triple_shape = triples.shape
    if dim == 1:
        triples = triples.t()
    new_cols = []
    for c in range(triples.shape[1]):
        this_column = triples[:, c]
        if c in columns:
            new_cols.append(torch.tensor([compaction[int(e)] for e in this_column]))
        else:
            new_cols.append(this_column)
    new_triples = torch.stack(new_cols, dim=1 - dim)
    assert new_triples.shape == triple_shape
    return new_triples


def compact_columns(
    triples: torch.LongTensor,
    label_to_id_mapping: Mapping[str, int],
    columns: Union[int, Collection[int]],
    dim=0,
) -> Tuple[torch.LongTensor, Optional[Mapping[str, int]], Optional[Mapping[int, int]]]:
    ids = label_to_id_mapping.values()
    num_ids = len(ids)
    assert len(set(ids)) == len(ids)
    max_id = max(ids)
    if num_ids < max_id + 1:
        compaction = dict((old, new) for new, old in enumerate(sorted(ids)))
        assert set(compaction.keys()) == set(label_to_id_mapping.values())
        assert set(compaction.values()) == set(range(num_ids))
        new_triples = apply_compaction(triples, compaction, columns, dim=dim)
        new_mapping = {label: compaction[_id] for label, _id in label_to_id_mapping.items()}
        logging.info(f'Compacted: {max_id} -> {num_ids - 1}')
    else:
        compaction = None
        new_triples = triples
        new_mapping = label_to_id_mapping
        logging.info(f'No compaction necessary.')
    return new_triples, new_mapping, compaction


def compact_graph(graph: KnowledgeGraph, no_duplicates: bool = True) -> Tuple[KnowledgeGraph, Optional[Mapping[int, int]], Optional[Mapping[int, int]]]:
    if graph.inverse_triples:
        raise NotImplementedError

    triples0 = graph.triples

    # Compact entities
    triples1, compact_entity_label_to_id, entity_compaction = compact_columns(triples=triples0, label_to_id_mapping=graph.entity_label_to_id, columns=(0, 2))

    # Compact relations
    triples2, compact_relation_label_to_id, relation_compaction = compact_columns(triples=triples1, label_to_id_mapping=graph.relation_label_to_id, columns=(1,))

    # Filter duplicates
    if no_duplicates:
        old_size = triples2.shape[0]
        triples2 = torch.unique(triples2, dim=0)
        new_size = triples2.shape[0]
        if new_size < old_size:
            logging.info(f'Aggregated edges: {old_size} -> {new_size}.')

    # Compile to new knowledge graph
    compact_graph_ = KnowledgeGraph(
        triples=triples2,
        entity_label_to_id=compact_entity_label_to_id,
        relation_label_to_id=compact_relation_label_to_id,
        lang_code=graph.lang_code,
        dataset_name=graph.dataset_name,
        subset_name=graph.subset_name
    )

    return compact_graph_, entity_compaction, relation_compaction


def compact_single_alignment(
    single_alignment: torch.LongTensor,
    left_compaction: Mapping[int, int],
    right_compaction: Mapping[int, int],
) -> torch.LongTensor:
    compact_single_alignment_ = single_alignment
    for col, compaction in enumerate([left_compaction, right_compaction]):
        compact_single_alignment_ = apply_compaction(triples=compact_single_alignment_, compaction=compaction, columns=col, dim=1)
    return compact_single_alignment_


def compact_knowledge_graph_alignment(
    alignment: EntityAlignment,
    left_entity_compaction: Mapping[int, int],
    right_entity_compaction: Mapping[int, int],
) -> EntityAlignment:
    # Entity compaction
    compact_entity_alignment_train = compact_single_alignment(single_alignment=alignment.train, left_compaction=left_entity_compaction, right_compaction=right_entity_compaction)
    compact_entity_alignment_test = compact_single_alignment(single_alignment=alignment.test, left_compaction=left_entity_compaction, right_compaction=right_entity_compaction)

    return EntityAlignment(
        train=compact_entity_alignment_train,
        test=compact_entity_alignment_test,
    )


def compact_knowledge_graph_alignment_dataset(
    left_graph: KnowledgeGraph,
    right_graph: KnowledgeGraph,
    alignment: EntityAlignment,
    no_duplicates: bool = True,
) -> Tuple[KnowledgeGraph, KnowledgeGraph, EntityAlignment]:
    left_compact_graph, left_entity_compaction, _ = compact_graph(graph=left_graph, no_duplicates=no_duplicates)
    right_compact_graph, right_entity_compaction, _ = compact_graph(graph=right_graph, no_duplicates=no_duplicates)
    compact_alignment = compact_knowledge_graph_alignment(
        alignment=alignment,
        left_entity_compaction=left_entity_compaction,
        right_entity_compaction=right_entity_compaction,
    )
    return left_compact_graph, right_compact_graph, compact_alignment


class KnowledgeGraphAlignmentDataset:
    """A knowledge graph alignment data set."""
    #: The first knowledge graph
    left_graph: KnowledgeGraph

    #: The second knowledge graph
    right_graph: KnowledgeGraph

    #: The alignment
    alignment: EntityAlignment

    def __init__(
        self,
        left_graph: KnowledgeGraph,
        right_graph: KnowledgeGraph,
        alignment: EntityAlignment,
    ):
        self.left_graph = left_graph
        self.right_graph = right_graph
        self.alignment = alignment

    def compact(self) -> 'KnowledgeGraphAlignmentDataset':
        left_graph, right_graph, alignment = compact_knowledge_graph_alignment_dataset(left_graph=self.left_graph, right_graph=self.right_graph, alignment=self.alignment)
        return KnowledgeGraphAlignmentDataset(
            left_graph=left_graph,
            right_graph=right_graph,
            alignment=alignment,
        )

    def with_inverse_triples(self) -> 'KnowledgeGraphAlignmentDataset':
        return KnowledgeGraphAlignmentDataset(
            left_graph=self.left_graph.with_inverse_triples(),
            right_graph=self.right_graph.with_inverse_triples(),
            alignment=self.alignment,
        )

    def with_self_loops(self) -> 'KnowledgeGraphAlignmentDataset':
        return KnowledgeGraphAlignmentDataset(
            left_graph=self.left_graph.with_self_loops(),
            right_graph=self.right_graph.with_self_loops(),
            alignment=self.alignment,
        )

    def validation_split(self, train_ratio: float, seed: Optional[int] = None) -> 'KnowledgeGraphAlignmentDataset':
        return KnowledgeGraphAlignmentDataset(
            left_graph=self.left_graph,
            right_graph=self.right_graph,
            alignment=self.alignment.validation_split(train_ratio=train_ratio, seed=seed),
        )

    @property
    def edge_tensors(self) -> Mapping[MatchSideEnum, torch.LongTensor]:
        return {
            MatchSideEnum.left: self.left_graph.edge_tensor_unique,
            MatchSideEnum.right: self.right_graph.edge_tensor_unique,
        }

    @property
    def graphs(self) -> Mapping[MatchSideEnum, KnowledgeGraph]:
        return {
            MatchSideEnum.left: self.left_graph,
            MatchSideEnum.right: self.right_graph,
        }

    @property
    def num_nodes(self) -> Mapping[MatchSideEnum, int]:
        return {
            MatchSideEnum.left: self.num_left_entities,
            MatchSideEnum.right: self.num_right_entities,
        }

    @property
    def left_triples(self) -> torch.LongTensor:
        return self.left_graph.triples

    @property
    def right_triples(self) -> torch.LongTensor:
        return self.right_graph.triples

    @property
    def entity_alignment_train(self) -> torch.LongTensor:
        return self.alignment.train

    @property
    def entity_alignment_test(self) -> torch.LongTensor:
        return self.alignment.test

    @property
    def num_left_triples(self) -> int:
        return self.left_graph.num_triples

    @property
    def num_left_entities(self) -> int:
        return self.left_graph.num_entities

    @property
    def num_left_relations(self) -> int:
        return self.left_graph.num_relations

    @property
    def num_right_triples(self) -> int:
        return self.right_graph.num_triples

    @property
    def num_right_entities(self) -> int:
        return self.right_graph.num_entities

    @property
    def num_right_relations(self) -> int:
        return self.right_graph.num_relations

    @property
    def num_train_alignments(self) -> int:
        return self.alignment.num_train

    @property
    def num_test_alignments(self) -> int:
        return self.alignment.num_test

    @property
    def num_exclusives(self) -> Mapping[MatchSideEnum, int]:
        return {
            side: num_nodes - (self.alignment.num_train + self.alignment.num_validation + self.alignment.num_test)
            for side, num_nodes in self.num_nodes.items()
        }

    @property
    def exclusives(self) -> Mapping[MatchSideEnum, torch.LongTensor]:
        return {
            side: torch.as_tensor(
                data=sorted(set(range(self.graphs[side].num_entities)).difference(aligned_on_side.tolist())),
                dtype=torch.long,
            )
            for side, aligned_on_side in zip(
                [MatchSideEnum.left, MatchSideEnum.right],
                self.alignment.all,
            )
        }

    @property
    def dataset_name(self) -> str:
        return self.left_graph.dataset_name

    @property
    def subset_name(self) -> str:
        return self.left_graph.subset_name

    def __str__(self):
        return f'{self.__class__.__name__}(left={self.left_graph}, right={self.right_graph}, align={self.alignment})'


class OnlineKnowledgeGraphAlignmentDatasetLoader:
    """Contains a lazy reference to a knowledge graph alignment data set."""

    #: The URL where the data can be downloaded from
    url: str

    #: The directory where the datasets will be extracted to
    cache_root: str

    def __init__(
        self,
        url: str,
        cache_root: Optional[str] = None,
        archive_file_name: Optional[str] = None,
        extractor: Extractor = TarExtractor(),
        **kwargs
    ) -> None:
        """Initialize the data set."""
        if archive_file_name is None:
            archive_file_name = url.rsplit(sep='/', maxsplit=1)[-1]
        self.archive_file_name = archive_file_name

        if cache_root is None:
            cache_root = tempfile.gettempdir()
        self.cache_root = path.join(cache_root, self.__class__.__name__.lower())

        self.url = url

        if extractor is None:
            extractor = TarExtractor()
        self.extractor = extractor

        if len(kwargs) > 0:
            logging.warning(f'Ignoring kwargs={kwargs}')

    def load(self, force_download: bool = False, force_extraction: bool = False) -> KnowledgeGraphAlignmentDataset:
        os.makedirs(self.cache_root, exist_ok=True)

        # Check if files already exist
        archive_path = path.join(self.cache_root, self.archive_file_name)
        if not path.isfile(archive_path) or force_download:
            if 'drive.google.com' in self.url:
                _id = self.url.split('?id=')[1]
                download_file_from_google_drive(id_=_id, destination=archive_path)
            else:
                logging.info(f'Requesting dataset from {self.url}')
                r = requests.get(url=self.url)
                assert r.status_code == requests.codes.ok  # pylint: disable=no-member
                with open(archive_path, 'wb') as archive_file:
                    archive_file.write(r.content)

        else:
            logging.info(f'Skipping to download from {self.url} due to existing files in {self.cache_root}.')

        # Extract files
        if force_download:
            force_extraction = True

        self.extractor.extract(archive_path=archive_path, cache_root=self.cache_root, force_extraction=force_extraction)

        # Load data
        left_graph, right_graph, alignment = self._load()
        logging.info(f'Loaded dataset.')

        return KnowledgeGraphAlignmentDataset(
            left_graph=left_graph,
            right_graph=right_graph,
            alignment=alignment,
        )

    def _load(self) -> Tuple[KnowledgeGraph, KnowledgeGraph, EntityAlignment]:
        left_graph = self._load_graph(left=True)
        right_graph = self._load_graph(left=False)
        alignment = self._load_alignment(left_graph=left_graph, right_graph=right_graph)
        return left_graph, right_graph, alignment

    def _load_graph(self, left: bool) -> KnowledgeGraph:
        raise NotImplementedError

    def _load_alignment(self, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        raise NotImplementedError


class _DBP15k(OnlineKnowledgeGraphAlignmentDatasetLoader, abc.ABC):
    SUBSETS = {'zh_en', 'ja_en', 'fr_en'}

    def __init__(
        self,
        url: str,
        subset: Optional[str] = None,
        cache_root: Optional[str] = None,
        train_test_split: Optional[float] = None,
        random_seed: int = 42,
        **kwargs
    ):
        if subset is None:
            subset = 'fr_en'
        if subset not in self.SUBSETS:
            raise KeyError(f'Unknown subset: {subset}. Allowed: {self.SUBSETS}.')
        self.subset = subset

        if train_test_split is None:
            train_test_split = 0.3
        if train_test_split <= 0. or train_test_split >= 1.:
            raise ValueError(f'Split must be a float with 0 < train_test_split < 1, but train_test_split={train_test_split},')
        self.split = train_test_split

        self.random_seed = random_seed

        super().__init__(
            url=url,
            cache_root=cache_root,
            **kwargs
        )


class DBP15kJAPE(_DBP15k):
    SPLITS = {str(i) for i in (10, 20, 30, 40, 50)}
    URL = 'https://github.com/nju-websoft/JAPE/raw/master/data/dbp15k.tar.gz'

    def __init__(
        self,
        subset: Optional[str] = None,
        cache_root: Optional[str] = None,
        train_test_split: Optional[float] = None,
        **kwargs
    ):
        allowed_splits = self.SPLITS.union([None])
        if train_test_split not in allowed_splits:
            raise ValueError(f'{self.__class__.__name__} only supports train_test_split in {allowed_splits}, but got {train_test_split}.')
        super().__init__(
            url=self.URL,
            subset=subset,
            cache_root=cache_root,
            train_test_split=train_test_split,
            **kwargs
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'dbp15k', self.subset, f'0_{str(int(100 * self.split))[0]}')

    def _load_graph(self, left: bool) -> KnowledgeGraph:
        lang_codes = self.subset.split('_')
        lang_code = lang_codes[0] if left else lang_codes[1]

        num = 1 if left else 2
        triple_path = path.join(self.root, f'triples_{num}')
        triples: torch.LongTensor = torch.tensor(
            data=[
                [int(_id) for _id in row]
                for row in _load_file(file_path=triple_path)
            ],
            dtype=torch.long,
        )
        id2e_path = path.join(self.root, f'ent_ids_{num}')
        entity_to_id = {entity: int(_id) for _id, entity in _load_file(id2e_path)}
        id2r_path = path.join(self.root, f'rel_ids_{num}')
        relation_to_id = {rel: int(_id) for _id, rel in _load_file(id2r_path)}

        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_to_id,
            relation_label_to_id=relation_to_id,
            lang_code=lang_code,
            dataset_name='dbp15kjape',
            subset_name=self.subset
        )

    def _load_alignment(self, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        ea_train, ea_test = [
            torch.tensor(
                data=[
                    [int(e) for e in row]
                    for row in _load_file(file_path=path.join(self.root, fp))
                ],
                dtype=torch.long,
            ).t()
            for fp in ['sup_ent_ids', 'ref_ent_ids']
        ]
        return EntityAlignment(
            train=ea_train,
            test=ea_test,
        )


class DBP15kFull(_DBP15k):
    URL = 'http://ws.nju.edu.cn/jape/data/DBP15k.tar.gz'

    def __init__(
        self,
        subset: Optional[str] = None,
        cache_root: Optional[str] = None,
        train_test_split: Optional[float] = None,
        random_seed: int = 42,
        **kwargs
    ):
        super().__init__(
            url=self.URL,
            subset=subset,
            cache_root=cache_root,
            train_test_split=train_test_split,
            random_seed=random_seed,
            **kwargs
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'DBP15k', self.subset)

    def _load_graph(self, left: bool) -> KnowledgeGraph:
        lang_codes = self.subset.split('_')
        lang_code = lang_codes[0] if left else lang_codes[1]

        triples_path = path.join(self.root, f'{lang_code}_rel_triples')

        triples, entity_label_to_id, relation_label_to_id = load_triples(triples_path=triples_path, delimiter='\t')
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
            lang_code=lang_code,
            dataset_name='dbp15kfull',
            subset_name=self.subset
        )

    def _load_alignment(self, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        entity_alignment_path = path.join(self.root, 'ent_ILLs')
        entity_alignment = _load_alignment(
            alignment_path=entity_alignment_path,
            left_label_to_id=left_graph.entity_label_to_id,
            right_label_to_id=right_graph.entity_label_to_id,
        )

        # Random split
        generator: torch.Generator = torch.Generator(device=entity_alignment.device).manual_seed(self.random_seed)
        num_alignments = entity_alignment.shape[1]
        perm = torch.randperm(num_alignments, generator=generator)
        last_train_idx = int(self.split * num_alignments)
        entity_alignment_train = entity_alignment.t()[perm[:last_train_idx]].t()
        entity_alignment_test = entity_alignment.t()[perm[last_train_idx:]].t()

        return EntityAlignment(
            train=entity_alignment_train,
            test=entity_alignment_test,
        )


class DWY100k(OnlineKnowledgeGraphAlignmentDatasetLoader):
    SUBSETS = {'wd', 'yg'}
    URL = 'https://drive.google.com/open?id=1AvLxawvI7J0oFhCtp2il7j7bBUDonBbr'

    def __init__(
        self,
        subset: Optional[str] = None,
        cache_root: Optional[str] = None,
        train_test_split: Optional[float] = None,
        random_seed: int = 42,
        **kwargs
    ):
        if subset is None:
            subset = 'wd'
        if subset not in DWY100k.SUBSETS:
            raise KeyError(f'Unknown subset: {subset}. Allowed: {DWY100k.SUBSETS}.')
        self.subset = subset

        if train_test_split is None:
            train_test_split = 0.3
        if train_test_split != 0.3:
            raise ValueError(f'Illegal train_test_split for DWY100k')
        self.split = train_test_split

        self.random_seed = random_seed

        super().__init__(
            url=DWY100k.URL,
            cache_root=cache_root,
            extractor=ZipExtractor(),
            **kwargs
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'DWY100K', f'dbp_{self.subset}', 'mapping', '0_3')

    def _load_graph(self, left: bool) -> KnowledgeGraph:
        num = 1 if left else 2
        triple_path = path.join(self.root, f'triples_{num}')
        triples: torch.LongTensor = torch.tensor(
            data=[
                [int(_id) for _id in row]
                for row in _load_file(file_path=triple_path)
            ],
            dtype=torch.long,
        )
        id2e_path = path.join(self.root, f'ent_ids_{num}')
        entity_to_id = {entity: int(_id) for _id, entity in _load_file(id2e_path)}
        id2r_path = path.join(self.root, f'rel_ids_{num}')
        relation_to_id = {rel: int(_id) for _id, rel in _load_file(id2r_path)}

        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_to_id,
            relation_label_to_id=relation_to_id,
            dataset_name='dwy100k',
            subset_name=self.subset,
        )

    def _load_alignment(self, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        ea_train, ea_test = [
            torch.tensor(
                data=[
                    [int(e) for e in row]
                    for row in _load_file(file_path=path.join(self.root, fp))
                ],
                dtype=torch.long,
            ).t()
            for fp in ['sup_ent_ids', 'ref_ent_ids']
        ]
        return EntityAlignment(
            train=ea_train,
            test=ea_test,
        )


class _WK3l(OnlineKnowledgeGraphAlignmentDatasetLoader, abc.ABC):
    URL = 'https://drive.google.com/open?id=1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z'
    SUBSETS = {'en_de', 'en_fr'}
    SIZES = {'15k', '120k'}

    def __init__(
        self,
        cache_root: Optional[str] = None,
        subset: Optional[str] = None,
        size: Optional[str] = None,
        train_test_split: Optional[float] = None,
        **kwargs
    ):
        if subset is None:
            subset = 'en_de'
        if subset not in _WK3l.SUBSETS:
            raise KeyError(f'Unknown subset: {subset}. Allowed are: {_WK3l.SUBSETS}.')
        self.subset = subset

        if size is None:
            size = '15k'
        if size not in _WK3l.SIZES:
            raise KeyError(f'Unknown size: {size}. Allowed are: {_WK3l.SIZES}.')
        self.size = size

        if train_test_split is None:
            train_test_split = .3
        self.split = train_test_split

        super().__init__(
            url=_WK3l.URL,
            cache_root=cache_root,
            extractor=ZipExtractor(),
            archive_file_name='wk3l',
            **kwargs
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'data', f'WK3l-{self.size}', self.subset)

    def _load_graph(self, left: bool) -> KnowledgeGraph:
        lang_match, lang_ref = self.subset.split('_')
        lang = lang_match if left else lang_ref
        version = 5 if self.subset == 'en_fr' else 6
        suffix = f'{version}' if self.size == '15k' else f'{version}_{self.size}'
        triples_path = path.join(self.root, f'P_{lang}_v{suffix}.csv')
        triples, entity_label_to_id, relation_label_to_id = load_triples(triples_path=triples_path, delimiter='@@@')
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
            lang_code=lang_match if left else lang_ref,
            dataset_name=f'wk3l_{self.size}',
            subset_name=self.subset,
        )

    def _load_alignment(self, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        # From first ILLs
        lang_a, lang_b = self.subset.split('_')
        suffix = '' if self.size == '15k' else '_' + str(self.size)
        first_ill_path = path.join(self.root, f'{lang_a}2{lang_b}_fk{suffix}.csv')
        first_ill_label_alignment = _load_file(file_path=first_ill_path, delimiter='@@@')
        first_ill_id_alignment = _label_alignment_to_id_alignment(
            first_ill_label_alignment,
            left_graph.entity_label_to_id,
            right_graph.entity_label_to_id,
        )
        logging.info(f'Loaded alignment of size {len(first_ill_id_alignment)} from first ILL: {first_ill_path}.')

        # From second ILLs
        second_ill_path = path.join(self.root, f'{lang_b}2{lang_a}_fk{suffix}.csv')
        second_ill_label_alignment = _load_file(file_path=second_ill_path, delimiter='@@@')
        second_ill_id_alignment = _label_alignment_to_id_alignment(
            second_ill_label_alignment,
            right_graph.entity_label_to_id,
            left_graph.entity_label_to_id,
        )
        second_ill_id_alignment = set(tuple(reversed(a)) for a in second_ill_id_alignment)
        logging.info(f'Loaded alignment of size {len(second_ill_id_alignment)} from second ILL: {second_ill_path}.')

        # Load label alignment
        version = 5 if self.subset == 'en_fr' else 6
        suffix = f'{version}' if self.size == '15k' else f'{version}_{self.size}'
        triple_alignment_path = path.join(self.root, f'P_{self.subset}_v{suffix}.csv')
        triple_alignment = _load_file(file_path=triple_alignment_path, delimiter='@@@')

        # From triples
        subject_alignment = set((row[0], row[3]) for row in triple_alignment)
        object_alignment = set((row[2], row[5]) for row in triple_alignment)
        entity_alignment = subject_alignment.union(object_alignment)
        triple_id_alignment = _label_alignment_to_id_alignment(
            entity_alignment,
            left_graph.entity_label_to_id,
            right_graph.entity_label_to_id,
        )
        logging.info(
            f'Loaded alignment of size {len(triple_id_alignment)} from triple alignment: {triple_alignment_path}.')

        # Merge alignments
        id_alignment = first_ill_id_alignment.union(second_ill_id_alignment).union(triple_id_alignment)
        logging.info(f'Merged alignments to alignment of size {len(id_alignment)}.')

        # As he split used by MTransE (ILL for testing, triples alignments for training) contains more than 95% test leakage, we use our own split
        sorted_id_alignment = numpy.asarray(sorted(id_alignment))
        assert sorted_id_alignment.shape[1] == 2
        rnd = numpy.random.RandomState(seed=42)  # noqa: E1101
        rnd.shuffle(sorted_id_alignment)
        split_idx = int(numpy.round(self.split * len(sorted_id_alignment)))
        entity_alignment_train: torch.LongTensor = torch.tensor(data=sorted_id_alignment[:split_idx, :].T, dtype=torch.long)
        entity_alignment_test: torch.LongTensor = torch.tensor(data=sorted_id_alignment[split_idx:, :].T, dtype=torch.long)
        logging.info(
            f'Split alignments to {100 * self.split:2.2f}% train equal to size {entity_alignment_train.shape[1]},'
            f'and {100 * (1. - self.split):2.2f}% test equal to size {entity_alignment_test.shape[1]}.')

        alignment = EntityAlignment(
            train=entity_alignment_train,
            test=entity_alignment_test,
        )

        return alignment


class WK3l15k(_WK3l):
    def __init__(
        self,
        cache_root: Optional[str] = None,
        subset: Optional[str] = None,
        train_test_split: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            cache_root=cache_root,
            subset=subset,
            size='15k',
            train_test_split=train_test_split,
            **kwargs
        )


class WK3l120k(_WK3l):
    def __init__(
        self,
        cache_root: Optional[str] = None,
        subset: Optional[str] = None,
        train_test_split: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            cache_root=cache_root,
            subset=subset,
            size='120k',
            train_test_split=train_test_split,
            **kwargs
        )


def dataset_name_normalization(name: str) -> str:
    return name.lower().replace('_', '')


def available_datasets() -> Mapping[str, Collection[str]]:
    """List available datasets with their subsets."""
    return {
        dataset_name_normalization(cls.__name__): cls.SUBSETS
        for cls in get_all_subclasses(base_class=OnlineKnowledgeGraphAlignmentDatasetLoader)
        if hasattr(cls, 'SUBSETS') and not cls.__name__.startswith('_')
    }


def get_dataset_by_name(
    dataset_name: str,
    subset_name: Optional[str] = None,
    cache_root: Optional[str] = None,
    inverse_triples: bool = False,
    self_loops: bool = False,
    compact: bool = True,
    train_test_split: Union[None, str, float] = None,
    train_validation_split: Optional[float] = None,
    force_download: bool = False,
    force_extraction: bool = False,
    random_seed: Optional[int] = 42,
    **kwargs
) -> KnowledgeGraphAlignmentDataset:
    """Load a dataset specified by name and subset name.

    :param dataset_name:
        The case-insensitive dataset name. One of ("DBP15k", )
    :param subset_name:
        An optional subset name
    :param cache_root:
        An optional cache directory for extracted downloads. If None is given, use /tmp/{dataset_name}
    :param inverse_triples:
        Whether to generate inverse triples (o, p_inv, s) for every triple (s, p, o).
    :param self_loops:
        Whether to generate self-loops (e, self_loop, e) for each entity e.
    :param compact:
        Whether to apply compaction, i.e. ensure consecutive relation and entity IDs.
    :param train_test_split:
        A specification of the train-test split to use.
    :param force_download:
        Force downloading the files even if they already exist.
    :param force_extraction:
        Force archive extraction even if it was already extracted before.
    :param kwargs:
        Additional key-word based arguments passed to the individual datasets.

    :return: A dataset.
    """
    # Normalize train-test-split
    if train_test_split is None:
        train_test_split = 0.3
    if isinstance(train_test_split, str):
        train_test_split = int(train_test_split) / 100.
    assert isinstance(train_test_split, float)

    # Resolve data set loader class
    dataset_loader_cls = get_subclass_by_name(
        base_class=OnlineKnowledgeGraphAlignmentDatasetLoader,
        name=dataset_name,
        normalizer=dataset_name_normalization,
        exclude=_WK3l,
    )

    # Instantiate dataset loader
    dataset_loader = dataset_loader_cls(
        subset=subset_name,
        cache_root=cache_root,
        train_test_split=train_test_split,
        random_seed=random_seed,
        **kwargs
    )

    # load dataset
    dataset = dataset_loader.load(force_download=force_download, force_extraction=force_extraction)
    logging.info(f'Created dataset: {dataset}')

    if compact:
        dataset = dataset.compact()
        logging.info('Applied compaction.')

    if inverse_triples:
        dataset = dataset.with_inverse_triples()
        logging.info(f'Created inverse triples: {dataset}')

    if self_loops:
        dataset = dataset.with_self_loops()
        logging.info(f'Created self-loops: {dataset}')

    if train_validation_split is not None:
        dataset = dataset.validation_split(train_ratio=train_validation_split, seed=random_seed)
        logging.info(f'Train-Validation-Split: {dataset}')

    return dataset


def _load_alignment(
    alignment_path: Union[str, TextIO],
    left_label_to_id: Mapping[str, int],
    right_label_to_id: Mapping[str, int],
    delimiter: str = '\t',
    encoding: str = 'utf8',
) -> torch.LongTensor:
    # Load label alignment
    label_alignment = _load_file(file_path=alignment_path, delimiter=delimiter, encoding=encoding)

    alignment = _label_alignment_to_id_alignment(
        label_alignment,
        left_label_to_id,
        right_label_to_id,
    )
    alignment = torch.tensor(list(zip(*alignment)), dtype=torch.long)

    logging.info(f'Loaded alignment of size {alignment.shape[1]}')

    return alignment


def _label_alignment_to_id_alignment(
    label_array: Collection[Collection[str]],
    *column_label_to_ids: Mapping[str, int],
) -> Set[Collection[int]]:
    num_raw = len(label_array)

    # Drop duplicates
    label_array = set(map(tuple, label_array))
    num_without_duplicates = len(label_array)
    if num_without_duplicates < num_raw:
        logging.warning(f'Dropped {num_raw - num_without_duplicates} duplicate rows.')

    # Translate to id
    result = {
        tuple(l2i.get(e, None) for l2i, e in zip(column_label_to_ids, row))
        for row in label_array
    }
    before_filter = len(result)
    result = set(row for row in result if None not in row)
    after_filter = len(result)
    logging.info(f'Translated list of length {before_filter} to label array of length {after_filter}.')

    return result


def load_triples(
    triples_path: Union[str, TextIO],
    delimiter: str = '\t',
    encoding: str = 'utf8',
) -> Tuple[torch.LongTensor, Mapping[str, int], Mapping[str, int]]:
    """Load triples."""
    # Load triples from tsv file
    label_triples = _load_file(file_path=triples_path, delimiter=delimiter, encoding=encoding)

    # Split
    heads, relations, tails = [[t[i] for t in label_triples] for i in range(3)]

    # Sorting ensures consistent results when the triples are permuted
    entity_label_to_id = {
        e: i for i, e in enumerate(sorted(set(heads).union(tails)))
    }
    relation_label_to_id = {
        r: i for i, r in enumerate(sorted(set(relations)))
    }

    id_triples = _label_alignment_to_id_alignment(
        label_triples,
        entity_label_to_id,
        relation_label_to_id,
        entity_label_to_id,
    )
    triples: torch.LongTensor = torch.tensor(data=list(id_triples), dtype=torch.long)

    # Log some info
    num_triples = triples.shape[0]
    num_entities = len(entity_label_to_id)
    num_relations = len(relation_label_to_id)
    logging.info(f'Loaded {num_triples} unique triples, '
                 f'with {num_entities} unique entities, '
                 f'and {num_relations} unique relations.')

    return triples, entity_label_to_id, relation_label_to_id


def _load_file(
    file_path: str,
    delimiter: str = '\t',
    encoding: str = 'utf8',
) -> List[List[str]]:
    with open(file_path, 'r', encoding=encoding) as f:
        out = [line[:-1].split(sep=delimiter) for line in f.readlines()]
    return out


def validation_split(
    alignment: EntityAlignment,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> EntityAlignment:
    # Check input
    if not (0. < train_ratio < 1.):
        raise ValueError(f'train_ratio must be between 0 and 1, but is {train_ratio}')

    # random seeding
    if seed is not None:
        generator = torch.manual_seed(seed=seed)
    else:
        generator = torch.random.default_generator

    # re-combine train and validation, if already split
    num_total = alignment.num_train
    pool = alignment.train
    if alignment.num_validation > 0:
        num_total += alignment.num_validation
        pool = torch.cat([pool, alignment.validation], dim=1)

    # reproducibility: sort
    indices = torch.argsort(pool[0, :])
    pool = pool[:, indices]
    indices = torch.argsort(pool[1, :])
    pool = pool[:, indices]

    # shuffle
    indices = torch.randperm(num_total, generator=generator, device=pool.device)
    pool = pool[:, indices]

    # split
    split_idx = int(train_ratio * num_total)
    train_alignments: torch.LongTensor = pool[:, :split_idx].detach().clone()
    validation_alignments: torch.LongTensor = pool[:, split_idx:].detach().clone()

    return EntityAlignment(
        train=train_alignments,
        _validation=validation_alignments,
        test=alignment.test,
    )


def exact_self_alignment(graph: KnowledgeGraph, train_percentage: float = 0.5) -> KnowledgeGraphAlignmentDataset:
    """
    Create a alignment between a graph a randomly permuted version of it.

    :param graph: The graph.
    :param train_percentage: The percentage of training alignments.

    :return: A knowledge graph alignment dataset.
    """
    # Create a random permutation as alignment
    full_alignment = torch.stack([
        torch.arange(graph.num_entities, dtype=torch.long),
        torch.randperm(graph.num_entities)
    ], dim=0)

    # shuffle
    full_alignment = full_alignment[:, torch.randperm(graph.num_entities)]

    # create mapping
    mapping = {int(a): int(b) for a, b in full_alignment.t()}

    # translate triples
    h, r, t = graph.triples.t()
    h_new, t_new = [torch.tensor([mapping[int(e)] for e in es], dtype=torch.long) for es in (h, t)]
    r_new = r.detach().clone()
    new_triples: torch.LongTensor = torch.stack([h_new, r_new, t_new], dim=-1)

    # compose second KG
    second_graph = KnowledgeGraph(
        triples=new_triples,
        entity_label_to_id={k: mapping[v] for k, v in graph.entity_label_to_id.items()},
        relation_label_to_id=graph.relation_label_to_id.copy(),
        inverse_triples=False,
        self_loops=False,
    )
    second_graph.inverse_triples = graph.inverse_triples
    second_graph.self_loops = graph.self_loops

    # split alignment
    split_id = int(train_percentage * graph.num_entities)
    alignment = EntityAlignment(
        train=full_alignment[:, :split_id],
        test=full_alignment[:, split_id:],
    )

    return KnowledgeGraphAlignmentDataset(
        left_graph=graph,
        right_graph=second_graph,
        alignment=alignment,
    )


def sub_graph_alignment(
    graph: KnowledgeGraph,
    overlap: float = 0.5,
    ratio: float = 0.7,
    train_test_split: float = 0.5,
) -> KnowledgeGraphAlignmentDataset:
    # split entities
    num_entities = graph.num_entities
    split = torch.randperm(num_entities)
    num_overlap = int(overlap * num_entities)
    common = split[:num_overlap]
    left = num_overlap + int((num_entities - num_overlap) * ratio)
    left, right = split[num_overlap:left], split[left:]
    left = torch.cat([common, left])
    right = torch.cat([common, right])

    # create alignment
    alignment = torch.arange(num_overlap).unsqueeze(dim=0).repeat(2, 1)
    assert alignment.shape == (2, num_overlap)
    alignment = alignment[:, torch.randperm(num_overlap)]
    train_test_split = int(train_test_split * num_overlap)
    train, test = alignment[:, :train_test_split], alignment[:, train_test_split:]
    alignment = EntityAlignment(train=train, test=test)

    # induced subgraph
    graphs = []
    for ent in [left, right]:
        ent = set(ent.tolist())
        entity_label_to_id = {
            str(old_id): new_id
            for new_id, old_id in enumerate(ent)
        }
        triples = torch.as_tensor(data=[
            (entity_label_to_id[str(h)], r, entity_label_to_id[str(t)])
            for h, r, t in graph.triples.tolist()
            if (h in ent and t in ent)
        ], dtype=torch.long)
        graphs.append(KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=graph.relation_label_to_id,
        ))

    return KnowledgeGraphAlignmentDataset(
        left_graph=graphs[0],
        right_graph=graphs[1],
        alignment=alignment,
    )


def get_erdos_renyi(num_entities: int, num_relations: int, p: float = .1) -> KnowledgeGraph:
    graph = nx.erdos_renyi_graph(n=num_entities, p=p, directed=True)
    if len(graph.edges) > 0:
        head, tail = [torch.as_tensor(e, dtype=torch.long) for e in zip(*graph.edges)]
    else:
        logging.warning('Generated empty Erdos-Renyi graph.')
        head = tail = torch.empty(0, dtype=torch.long)
    relation = torch.randint(num_relations, size=(head.shape[0],))
    triples = torch.stack([head, relation, tail], dim=1)
    return KnowledgeGraph(
        triples=triples,
        entity_label_to_id={str(i): i for i in range(num_entities)},
        relation_label_to_id={str(i): i for i in range(num_relations)},
        dataset_name='erdos_renyi',
        subset_name=f'{num_entities}-{num_relations}-{p}',
    )


def get_synthetic_math_graph(
    num_entities: int,
) -> KnowledgeGraph:
    entities = list(range(num_entities))
    relations = list(range(num_entities))
    triples = [(e, r, (e + r) % num_entities) for r in relations for e in entities]
    return KnowledgeGraph(
        triples=torch.as_tensor(triples, dtype=torch.long),
        entity_label_to_id={str(e): e for e in entities},
        relation_label_to_id={'+' + str(r): r for r in relations},
    )


def download_file_from_google_drive(id_, destination):
    # cf. https://stackoverflow.com/a/39225272
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id_}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id_, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
