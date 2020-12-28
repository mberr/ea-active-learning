# coding=utf-8
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import torch
import tqdm
from torch import optim
from torch.optim import Adam, Optimizer

from kgm.data import MatchSideEnum
from kgm.data.edge_modifiers import EdgeModifierHeuristic
from kgm.eval import evaluate_alignment
from kgm.models import KGMatchingModel
from kgm.modules import MatchingLoss, SampledMatchingLoss, Similarity
from kgm.utils.common import get_subclass_by_name

_LOGGER = logging.getLogger(name=__name__)


class AlignmentModelTrainer:
    """
    A wrapper around a model encapsulating training and evaluation.
    """

    #: The model instance
    model: KGMatchingModel

    #: The similarity instance
    similarity: Similarity

    #: The loss instance
    loss: MatchingLoss

    #: The optimizer instance
    optimizer: Optimizer

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        num_epochs: Optional[int] = None,
        eval_frequency: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        loss: Optional[MatchingLoss] = None,
        loss_cls: Optional[Type[MatchingLoss]] = None,
        loss_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_cls: Type[Optimizer] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        edge_modifiers: Optional[Union[EdgeModifierHeuristic, List[EdgeModifierHeuristic]]] = None,
    ):
        self.model = model

        self.similarity = similarity

        if num_epochs is None:
            num_epochs = 100
        self.num_epochs = num_epochs

        if eval_frequency is None:
            eval_frequency = num_epochs
        self.eval_frequency = eval_frequency

        if eval_batch_size is None:
            eval_batch_size = 1
            _LOGGER.warning(f'Falling back to evaluation batch size of 1. This might result in inferior performance, and not be intended.')
        self.eval_batch_size = eval_batch_size

        if loss is not None:
            self.loss = loss
            if loss_cls is None or loss_kwargs is None:
                _LOGGER.warning('Ignoring loss_cls and loss_kwargs, as loss instance is passed.')
        else:
            # create loss instance
            if loss_cls is None:
                loss_cls = SampledMatchingLoss
            if loss_kwargs is None:
                loss_kwargs = {}
            self.loss = loss_cls(similarity=similarity, **loss_kwargs)

        # create optimizer
        if optimizer_cls is None:
            optimizer_cls = Adam
        if isinstance(optimizer_cls, str):
            optimizer_cls = get_subclass_by_name(base_class=optim.Optimizer, name=optimizer_cls)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer_cls(params=filter(lambda p: p.requires_grad, model.parameters()), **optimizer_kwargs)

        # create edge modifier
        if edge_modifiers is None:
            edge_modifiers = []
        if not isinstance(edge_modifiers, list):
            edge_modifiers = [edge_modifiers]
        self.edge_modifiers = edge_modifiers

    def prepare_for_training(
        self,
        edge_tensors: Mapping[MatchSideEnum, torch.LongTensor],
        train_alignment: torch.LongTensor,
        exclusives: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
    ) -> Tuple[torch.LongTensor, Mapping[MatchSideEnum, torch.LongTensor]]:
        # set in train mode
        self.model.train()

        # Modify edges
        for edge_modifier in self.edge_modifiers:
            edge_tensors = edge_modifier.modify_edges(edge_tensors=edge_tensors, alignment=train_alignment, exclusives=exclusives)

        # Clone and detach
        edge_tensors = {
            side: edge_tensor.detach().clone()
            for side, edge_tensor in edge_tensors.items()
        }

        # Set edge tensors
        self.model.set_edge_tensors_(edge_tensors=edge_tensors)

        # Determine loss candidates based on exclusives
        if exclusives is not None:
            candidates = {
                side: torch.tensor(
                    data=sorted(set(edge_tensor.unique()).difference(exclusives[side])),
                    dtype=torch.long,
                    device=self.device,
                )
                for side, edge_tensor in edge_tensors.items()
            }
        else:
            candidates = dict()

        # send to device
        train_alignment: torch.LongTensor = train_alignment.to(device=self.device)
        candidates: Mapping[MatchSideEnum, torch.LongTensor] = {
            side: this_side_candidates.to(self.device)
            for side, this_side_candidates in candidates.items()
        }

        return train_alignment, candidates

    def train(
        self,
        edge_tensors: Mapping[MatchSideEnum, torch.LongTensor],
        train_alignment: torch.LongTensor,
        exclusives: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
        validation_alignment: Optional[torch.LongTensor] = None,
        keep_all: bool = False,
        use_tqdm: bool = False,
    ) -> List[Dict[str, Any]]:
        """Train the model with the given alignment."""
        ea_train, candidates = self.prepare_for_training(
            edge_tensors=edge_tensors,
            train_alignment=train_alignment,
            exclusives=exclusives,
        )

        all_results = []
        epoch_result = None
        if use_tqdm:
            epochs = tqdm.trange(self.num_epochs, unit='epoch', unit_scale=True, leave=False)
        else:
            epochs = range(self.num_epochs)
        for e in epochs:
            # training step
            epoch_result = {
                'epoch': e + 1,
                'train': self.train_step(training_alignment=ea_train, candidates=candidates),
            }

            # evaluate
            if (e + 1) % self.eval_frequency == 0:
                with torch.no_grad():
                    if validation_alignment is not None:
                        epoch_result['validation'] = self.eval_step(validation_alignment=validation_alignment)
                        if self.should_stop(epoch_result):
                            break
            if keep_all:
                all_results.append(epoch_result)
        if not keep_all:
            all_results = [epoch_result]

        if validation_alignment is not None:
            with torch.no_grad():
                all_results[-1]['validation'] = self.eval_step(validation_alignment=validation_alignment)

        return all_results

    def should_stop(self, evaluation: Dict[str, Any]) -> bool:  # pylint: disable=unused-argument
        return False

    def eval_step(self, validation_alignment: torch.LongTensor) -> float:
        # Set in evaluation mode
        self.model.eval()

        # evaluate
        evaluation = evaluate_alignment(
            similarity=self.similarity,
            alignment=validation_alignment,
            representations=self.model(),
            eval_batch_size=self.eval_batch_size,
        )

        return evaluation

    def train_step(
        self,
        training_alignment: torch.LongTensor,
        candidates: Mapping[MatchSideEnum, Optional[torch.LongTensor]],
    ) -> float:
        """Perform a single training step and return loss value."""
        # Set to training mode
        self.model.train()

        # Get representations
        node_repr = self.model()

        # compute loss
        train_loss = self.loss.forward(alignment=training_alignment, representations=node_repr, candidates=candidates)

        # compute gradient
        train_loss.backward()

        # Parameter update
        self.optimizer.step()

        # zero gradient
        self.optimizer.zero_grad()

        # return training loss
        return train_loss.item()

    def evaluate(self, alignment: torch.LongTensor) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        with torch.no_grad():
            return evaluate_alignment(
                similarity=self.similarity,
                alignment=alignment,
                representations=self.model(),
                eval_batch_size=self.eval_batch_size,
            )

    @property
    def device(self) -> torch.device:
        """The model's device."""
        return self.model.device


class EarlyStoppingTrainer(AlignmentModelTrainer):
    """A wrapper that trains with early stopping."""

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        num_epochs: Optional[int] = None,
        eval_frequency: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        loss: Optional[MatchingLoss] = None,
        loss_cls: Optional[Type[MatchingLoss]] = None,
        loss_kwargs: Optional[Mapping[str, Any]] = None,
        optimizer_cls: Type[Optimizer] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        patience: int = 1,
        minimum_relative_difference: float = 0.,
        eval_key: str = 'hits_at_1',
        larger_is_better: bool = True,
        edge_modifiers: Optional[Union[EdgeModifierHeuristic, List[EdgeModifierHeuristic]]] = None,
    ):
        super().__init__(
            model=model,
            similarity=similarity,
            num_epochs=num_epochs,
            eval_frequency=eval_frequency,
            eval_batch_size=eval_batch_size,
            loss=loss,
            loss_cls=loss_cls,
            loss_kwargs=loss_kwargs,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            edge_modifiers=edge_modifiers,
        )
        self.max_patience = patience
        self.minimum_relative_difference = minimum_relative_difference
        self.eval_key = eval_key
        self.larger_is_better = larger_is_better
        self.best_value = None
        self.patience = self.max_patience

    def is_better(self, last_value: float, best_value: Optional[float]) -> bool:
        if best_value is None:
            return True
        if self.larger_is_better:
            return last_value > best_value * (1. + self.minimum_relative_difference)
        else:
            return last_value < best_value * (1. - self.minimum_relative_difference)

    def should_stop(self, evaluation: Dict[str, Any]) -> bool:
        last_value = evaluation['validation'][self.eval_key]
        # logging.info(f'ES: last={last_value}, best={self.best_value}, patience={self.patience}')
        if self.is_better(last_value=last_value, best_value=self.best_value):
            self.patience = self.max_patience
            self.best_value = last_value
        else:
            self.patience -= 1

        return self.patience < 0

    def train(
        self,
        edge_tensors: Mapping[MatchSideEnum, torch.LongTensor],
        train_alignment: torch.LongTensor,
        exclusives: Optional[Mapping[MatchSideEnum, torch.LongTensor]] = None,
        validation_alignment: Optional[torch.LongTensor] = None,
        keep_all: bool = False,
        use_tqdm: bool = False,
    ) -> List[Dict[str, Any]]:  # noqa: D102
        assert validation_alignment is not None
        self.best_value, self.patience = None, self.max_patience
        return super().train(
            edge_tensors=edge_tensors,
            train_alignment=train_alignment,
            exclusives=exclusives,
            validation_alignment=validation_alignment,
            keep_all=keep_all,
            use_tqdm=use_tqdm,
        )
