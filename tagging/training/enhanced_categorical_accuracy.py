from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("enhanced_categorical_accuracy")
class EnhancedCategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise ConfigurationError(
                "Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)"
            )
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, seq_len, num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, seq_len, num_labels).
        mask : `torch.Tensor`, optional (default = None).
            A masking tensor of shape (batch_size, seq_len).
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        
        # 62 for deprels
        num_classes = predictions.size(-1)
        
       
        #print(gold_labels)
        print("gl 1 size", gold_labels.size())
        # shape : (batch * seq_len * num_labels)
        
        gold_labels = gold_labels.view(-1, gold_labels.size(-1))
        print("gl 2 size", gold_labels.size())
        print("gl 2", gold_labels)

        # shape : (batch * seq_len, num_classes)
        predictions = predictions.view((-1, num_classes))
        #print("preds size", predictions.size)
               

        
        #gold_labels = gold_labels.unsqueeze(-1)
        #print("gl 3 size", gold_labels.size())
        #print("gl 3", gold_labels)

        # shape : (batch * seq_len, num_labels) 
        #g = gold_labels.view(-1, gold_labels.size(-1))
        #flat_g = g.view(-1, g.size(-1))
        #print("flat g ", flat_g)
        
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                # shape (batch * seq_len, 1)
                # should be the index element of highest prediction
                top_k = predictions.max(-1)[1].unsqueeze(-1)
                print("top k", top_k)
            else:
                # shape (b * seq_len, top_k)
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]
                

            # This is of shape (batch_size, ..., top_k).
            #print("gold labes", gold_labels)
            
            # shape : (b * s * nbl, 1)
            #x = gold_labels.unsqueeze(-1)
            #print("x shape", x.shape)
            
            #print("g", g)
            
            # top_k : (batch * seq_len, 1)
            # g : (batch * seq_len, num_labels)
            # need to drop the 0s in num_labels
            
            #correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
            print("top k shape", top_k.shape)
            #print("g shape", g.shape)
            
            correct = top_k.eq(gold_labels).float()
            print("correct", correct)
        else:
            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel()).long(), gold_labels
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            print("c2", correct)
            self.total_count += mask.sum()
            print(self.total_count)
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()
        
        print("correct count", self.correct_count)
        print("total count", self.total_count)

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated accuracy.
        """
        print(self.total_count)
        if self.total_count > 1e-12:
            print("it is")
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            print("it is not")
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0