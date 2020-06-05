# based on: https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/attachment_scores.py

from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("enhanced_attachment_scores")
class EnhancedAttachmentScores(Metric):
    """
    Computes unlabeled precision, recall and f1.
    As the parser uses a sigmoid, initially it will predict numerous edges
    therefore making recall high but precision low.

    :precision: correct / system_total
    :recall: correct / gold_total
    :f1: 2 * correct / (system_total + gold_total)

    TODO: need to figure out how to mask punctuation which is ignored.

    # Parameters
    ignore_classes : `List[int]`, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None) -> None:
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0

        # we're interested in edges rather than words because a word can have multiple edges.
        self._num_gold_edges = 0.0
        self._num_pred_edges = 0.0

        self._ignore_classes: List[int] = ignore_classes or []

    def __call__(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        predicted_labeled_arcs: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        gold_labeled_arcs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        # Parameters
        predicted_indices : `torch.Tensor`, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        predicted_labeled_arcs : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_indices`.
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_labels`.
        mask : `torch.Tensor`, optional (default = None).
            A tensor of the same shape as `predicted_indices`.
        """
        unwrapped = self.unwrap_to_tensors(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = unwrapped

        # UAS
        for gold_edges, gold_edge_labels, predicted_edges, predicted_edge_labels in zip(gold_indices, gold_labels, predicted_indices, predicted_labels):
            self._num_gold_edges += len(gold_edges)
            self._num_pred_edges += len(predicted_edges)
            for predicted_edge in predicted_edges:
                if predicted_edge in gold_edges:
                    self._unlabeled_correct += 1.

        # LAS
        for gold_labeled_edges, predicted_labeled_edges in zip(gold_labeled_arcs, predicted_labeled_arcs):
            for predicted_labeled_edge in predicted_labeled_edges:
                if predicted_labeled_edge in gold_labeled_edges:
                    self._labeled_correct += 1.



    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated metrics as a dictionary.
        """
        unlabeled_precision = 0.0
        unlabeled_recall = 0.0
        unlabeled_f1 = 0.0

        labeled_precision = 0.0
        labeled_recall = 0.0
        labeled_f1 = 0.0


        if self._num_gold_edges > 0.0:
            # precision
            unlabeled_precision = self._unlabeled_correct / self._num_pred_edges
            labeled_precision = self._labeled_correct / self._num_pred_edges

            # recall
            unlabeled_recall = self._unlabeled_correct / self._num_gold_edges
            labeled_recall = self._labeled_correct / self._num_gold_edges

            # f1
            unlabeled_f1 = 2 * self._unlabeled_correct / (self._num_pred_edges + self._num_gold_edges)
            labeled_f1 = 2 * self._labeled_correct / (self._num_pred_edges + self._num_gold_edges)

        if reset:
            self.reset()
        return {
            "unlabeled_precision": unlabeled_precision,
            "unlabeled_recall": unlabeled_recall,
            "unlabeled_f1": unlabeled_f1,
            "labeled_precision": labeled_precision,
            "labeled_recall": labeled_recall,
            "labeled_f1": labeled_f1
        }

    @overrides
    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._num_gold_edges = 0.0
        self._num_pred_edges = 0.0
