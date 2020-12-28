import unittest

from kgm.active_learning import RandomHeuristic
from kgm.data import get_erdos_renyi
from kgm.data.knowledge_graph import sub_graph_alignment
from kgm.models import GCNAlign
from kgm.modules import DotProductSimilarity, SampledMatchingLoss, MarginLoss
from kgm.training.active_learning import evaluate_active_learning_heuristic
from kgm.training.matching import AlignmentModelTrainer


class Test(unittest.TestCase):
    step_size: int = 10
    num_entities: int = 61
    num_relations: int = 3
    num_epochs: int = 3

    def test_evaluate_active_learning_heuristic(self):
        dataset = sub_graph_alignment(graph=get_erdos_renyi(num_entities=self.num_entities, num_relations=self.num_relations, p=0.4))
        model = GCNAlign(num_nodes=dataset.num_nodes, embedding_dim=2)
        similarity = DotProductSimilarity()
        heuristic = RandomHeuristic()
        trainer = AlignmentModelTrainer(
            model=model,
            similarity=similarity,
            num_epochs=self.num_epochs,
            loss=SampledMatchingLoss(
                similarity=similarity,
                pairwise_loss=MarginLoss(),
            )
        )
        for i, (step, result) in enumerate(evaluate_active_learning_heuristic(
            dataset=dataset,
            model=model,
            similarity=similarity,
            heuristic=heuristic,
            trainer=trainer,
            step_size=self.step_size,
            eval_batch_size=2,
        ), start=1):
            assert step == i * self.step_size
            assert result['epoch'] == self.num_epochs
            assert result['total_epoch'] == i * self.num_epochs
