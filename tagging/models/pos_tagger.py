from typing import Dict, Optional, List, Any
import logging

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Dropout

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, InputVariationalDropout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from tagging.nn.util import nested_sequence_cross_entropy_with_logits
from tagging.nn.util import get_label_field_mask


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("pos_tagger")
class PosTagger(Model):
    """
    This ``PosTaggerMonolingual`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag for each token in the sequence.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    label_namespace : ``str``, optional (default=``pos``)
        The labels (pos tags) we are predicting.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 label_namespace: str = "head_tags",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder


        self.label_namespace_pos = "pos_tags"
        self.label_namespace_tags = "head_tags"
        self.num_classes_pos = self.vocab.get_vocab_size(self.label_namespace_pos)
        self.num_classes_tags = self.vocab.get_vocab_size(self.label_namespace_tags)
        
        
        logger.info(f"found num classes pos  : {self.num_classes_pos}")
        logger.info(f"found num classes tags  : {self.num_classes_tags}")
        
        self.encoder = encoder
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        
        self.pos_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes_pos))
        
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes_tags))
        
        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
            
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class tags of shape
            ``(batch_size, num_tokens, max_padding_dim)``.        
        
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
     
        size_pos_tags = pos_tags.size()
        size_head_tags = head_tags.size()
        # - INFO - tagging.models.pos_tagger - pos tags size : torch.Size([32, 22])
        logger.info(f"pos tags size : {size_pos_tags}")
        # - INFO - tagging.models.pos_tagger - head tags size : torch.Size([32, 22, 3])
        logger.info(f"head tags size : {size_head_tags}")
        
        embedded_text_input = self.text_field_embedder(words)
        batch_size, sequence_length, _ = embedded_text_input.size()
        
        mask = get_text_field_mask(words)
        
        #label_mask = get_label_field_mask(words)
        
        
        encoded_text = self.encoder(embedded_text_input, mask)
        encoded_text_size = encoded_text.size()
        # - INFO - tagging.models.pos_tagger - encoded text size : torch.Size([32, 22, 200])
        logger.info(f"encoded text size : {encoded_text_size}")

        pos_logits = self.pos_projection_layer(encoded_text)
        pos_logits_size = pos_logits.size()
        # - INFO - tagging.models.pos_tagger - pos logits size : torch.Size([32, 22, 16])
        logger.info(f"pos logits size : {pos_logits_size}")
                
        tag_logits = self.tag_projection_layer(encoded_text)
        tag_logits_size = tag_logits.size()
        # - INFO - tagging.models.pos_tagger - tag logits size : torch.Size([32, 22, 62])
        logger.info(f"tag logits size : {tag_logits_size}")
        
        
        # torch.Size([32, 22, 16]) -> torch.Size([704, 16])
        reshaped_pos_log_probs = pos_logits.view(-1, self.num_classes_pos)
        reshaped_pos_logits_size = reshaped_pos_log_probs.size()
        # - INFO - tagging.models.pos_tagger - reshaped pos logits size : torch.Size([704, 16])
        logger.info(f"reshaped pos logits size : {reshaped_pos_logits_size}")

        reshaped_tag_log_probs = tag_logits.view(-1, self.num_classes_tags)
        reshaped_tag_logits_size = reshaped_tag_log_probs.size()
        # - INFO - tagging.models.pos_tagger - reshaped tag logits size : torch.Size([704, 62])
        logger.info(f"reshaped tag logits size : {reshaped_tag_logits_size}")
        
        
        pos_class_probabilities = F.softmax(reshaped_pos_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes_pos])
    
        tag_class_probabilities = F.softmax(reshaped_tag_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes_tags])  
    

        output_dict = {"pos_logits": pos_logits, "pos_class_probabilities": pos_class_probabilities,
                       "tag_logits": tag_logits, "tag_class_probabilities": tag_class_probabilities                       
                       }

        if pos_tags is not None:
            #criterion = nn.CrossEntropyLoss()
            #pos_loss = criterion(pos_logits, pos_tags)
            pos_loss = sequence_cross_entropy_with_logits(pos_logits, pos_tags, mask)
            for metric in self.metrics.values():
                metric(pos_logits, pos_tags, mask.float())
            output_dict["loss"] = pos_loss
            
        print(head_tags.shape)    
        #print(head_tags)
        if head_tags is not None:
            #criterion = nn.MultiLabelSoftMarginLoss()            
            #RuntimeError: The size of tensor a (3) must match the size of tensor b (62) at non-singleton dimension 2
            tag_loss = nested_sequence_cross_entropy_with_logits(tag_logits, head_tags, mask)
#            for metric in self.metrics.values():
#                metric(tag_logits, head_tags, mask.float())
            
            #output_dict["loss"] = tag_loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]

#            # include ids, heads and tags in dictionary for next task/evaluation
#            output_dict["ids"] = [x["ids"] for x in metadata if "ids" in x]
#            output_dict["predicted_dependencies"] = [x["head_tags"] for x in metadata] 
#            output_dict["predicted_heads"] = [x["head_indices"] for x in metadata] 
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            pos_tags = [self.vocab.get_token_from_index(x, namespace="pos")
                    for x in argmax_indices]
            all_tags.append(pos_tags)
        output_dict['pos'] = all_tags
        return output_dict
    
#        all_predictions = output_dict['class_probabilities']
#        all_predictions = all_predictions.cpu().data.numpy()
#        if all_predictions.ndim == 3:
#            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
#        else:
#            predictions_list = [all_predictions]
#        all_tags = []
#        for predictions in predictions_list:
#            argmax_indices = numpy.argmax(predictions, axis=-1)
#            head_tags = [self.vocab.get_token_from_index(x, namespace="head_tags")
#                    for x in argmax_indices]
#            all_tags.append(head_tags)
#        output_dict['head_tags'] = all_tags
#        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
