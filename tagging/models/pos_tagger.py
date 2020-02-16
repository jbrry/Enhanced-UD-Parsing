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
from tagging.training.enhanced_attachment_scores import EnhancedAttachmentScores
from tagging.training.enhanced_categorical_accuracy import EnhancedCategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}

@Model.register("pos_tagger")
class PosTagger(Model):
    """
    This ``SequenceLabeler`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag for each token in the sequence.

    It is in the process of predicting potentially multiple tags for each token in  the sequence.
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
        Note: we now have multiple label_namespaces.
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

        # POS
        self.label_namespace_pos = "pos_tags"
        
        # Basic Dependencies
        self.label_namespace_tags = "head_tags"
        self.label_namespace_heads = "head_indices"

        # Enhanced Dependencies
        self.label_namespace_enhanced_tags = "enhanced_head_tags"
        self.label_namespace_enhanced_heads = "enhanced_head_indices"
        
        # Number of heads per token
        self.label_namespace_num_heads = "head_num_tags"
        

        self.num_classes_pos = self.vocab.get_vocab_size(self.label_namespace_pos)        
        self.num_classes_tags = self.vocab.get_vocab_size(self.label_namespace_tags)
        self.num_classes_enhanced_tags = self.vocab.get_vocab_size(self.label_namespace_enhanced_tags)
        
        self.num_classes_enhanced_heads = self.vocab.get_vocab_size(self.label_namespace_enhanced_heads)
        self.num_classes_num_heads = self.vocab.get_vocab_size(self.label_namespace_num_heads)
        
        
        tags = self.vocab.get_token_to_index_vocabulary("pos_tags")
        punctuation_tag_indices = {tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE}
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(f"Found POS tags corresponding to the following punctuation : {punctuation_tag_indices}. "
                    "Ignoring words with these POS tags for evaluation.")
                
        logger.info(f"found num classes pos  : {self.num_classes_pos}")
        logger.info(f"found num classes basic tags  : {self.num_classes_tags}")
        logger.info(f"found num classes enhanced tags  : {self.num_classes_enhanced_tags}")
        #logger.info(f"found num classes num heads  : {self.num_classes_num_heads}")
        
        self.encoder = encoder
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        
        # MLPs
        self.num_projection_layer = Linear(self.encoder.get_output_dim(), 5)
        
        # POS
        self.pos_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes_pos))
        
        # Tags
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes_tags))
        
        # Enhanced Tags
        self.enhanced_tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes_enhanced_tags))
        

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        self.metrics = {
                "pos_accuracy": CategoricalAccuracy(),
                "tag_accuracy": CategoricalAccuracy(),
                "enhanced_tag_accuracy": EnhancedCategoricalAccuracy()
                }
        
        #self._enhanced_attachment_scores = EnhancedAttachmentScores()
        self._enhanced_accuracy = EnhancedCategoricalAccuracy()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None,
                enhanced_head_tags: torch.LongTensor = None,
                enhanced_head_indices: torch.LongTensor = None,                
                num_heads: torch.LongTensor = None,
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
     
        embedded_text_input = self.text_field_embedder(words)
        batch_size, sequence_length, _ = embedded_text_input.size()
        
        mask = get_text_field_mask(words)        
        
        # shape (batch, seq_len, ?)
        encoded_text = self.encoder(embedded_text_input, mask)
        encoded_text_size = encoded_text.size()
        # - INFO - tagging.models.pos_tagger - encoded text size : torch.Size([32, 22, 200])
        logger.info(f"encoded text size : {encoded_text_size}")
        
        
        # Logits
        num_logits = self.num_projection_layer(encoded_text)

        pos_logits = self.pos_projection_layer(encoded_text)
                
        tag_logits = self.tag_projection_layer(encoded_text)
        
        enhanced_tag_logits = self.enhanced_tag_projection_layer(encoded_text)
        
        #reshaped_num_log_probs = num_logits.view(-1, self.num_classes_num_heads)
        
        # -> shape (batch * seq_len, num_classes)
        reshaped_pos_log_probs = pos_logits.view(-1, self.num_classes_pos)

        reshaped_tag_log_probs = tag_logits.view(-1, self.num_classes_tags)
        
        reshaped_enhanced_tag_log_probs = enhanced_tag_logits.view(-1, self.num_classes_enhanced_tags)
        
        
        #num_class_probabilities = F.softmax(num_logits, dim =-1)
    
        pos_class_probabilities = F.softmax(reshaped_pos_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes_pos])

        tag_class_probabilities = F.softmax(reshaped_tag_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes_tags])  

        enhanced_tag_class_probabilities = F.softmax(reshaped_enhanced_tag_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes_enhanced_tags])  
    
        output_dict = {
                    "pos_logits": pos_logits, "pos_class_probabilities": pos_class_probabilities,
                    "tag_logits": tag_logits, "tag_class_probabilities": tag_class_probabilities,
                    "enhanced_tag_logits": enhanced_tag_logits, "enhanced_tag_class_probabilities": enhanced_tag_class_probabilities
                    #"num_logits": num_logits, "num_class_probabilities": num_class_probabilities
                    }

        #if num_heads is not None:
        #    num_loss = sequence_cross_entropy_with_logits(num_logits, num_heads, mask)
        #    for metric in self.metrics.values():
         #       metric(num_logits, num_heads, mask.float())
            #output_dict["loss"] = pos_loss
            
            
        if pos_tags is not None:
            pos_loss = sequence_cross_entropy_with_logits(pos_logits, pos_tags, mask)
            for metric in self.metrics.values():
                metric(pos_logits, pos_tags, mask.float())
#            output_dict["loss"] = pos_loss

        if head_tags is not None:
            tag_loss = sequence_cross_entropy_with_logits(tag_logits, head_tags, mask)
            for metric in self.metrics.values():
                metric(tag_logits, head_tags, mask.float())
#            output_dict["loss"] = tag_loss

        if enhanced_head_tags is not None:
            enhanced_tag_loss = nested_sequence_cross_entropy_with_logits(enhanced_tag_logits, enhanced_head_tags, mask)          
            for metric in self.metrics.values():
                metric(enhanced_tag_logits, enhanced_head_tags, mask.float())                     
            output_dict["loss"] = enhanced_tag_loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['enhanced_tag_class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            head_tags = [self.vocab.get_token_from_index(x, namespace="enhanced_head_tags")
                    for x in argmax_indices]
            all_tags.append(head_tags)
        output_dict["enhanced_head_tags"] = all_tags
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
