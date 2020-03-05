from typing import Dict, Optional, Tuple, Any, List
import logging
import copy
from operator import itemgetter 

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import F1Measure
from tagging.training.enhanced_attachment_scores import EnhancedAttachmentScores

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("enhanced_parser")
class EnhancedParser(Model):
    """
    A Parser for arbitrary graph structures.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for arc tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    edge_prediction_threshold : ``int``, optional (default = 0.5)
        The probability at which to consider a scored edge to be 'present'
        in the decoded graph. Must be between 0 and 1.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 tag_representation_dim: int,
                 arc_representation_dim: int,
                 tag_feedforward: FeedForward = None,
                 arc_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 edge_prediction_threshold: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EnhancedParser, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.edge_prediction_threshold = edge_prediction_threshold
        if not 0 < edge_prediction_threshold < 1:
            raise ConfigurationError(f"edge_prediction_threshold must be between "
                                     f"0 and 1 (exclusive) but found {edge_prediction_threshold}.")

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = arc_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    arc_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim,
                                                     arc_representation_dim,
                                                     use_input_biases=True)

        num_labels = self.vocab.get_vocab_size("labels")
        self.head_tag_feedforward = tag_feedforward or \
                                        FeedForward(encoder_dim, 1,
                                                    tag_representation_dim,
                                                    Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = BilinearMatrixAttention(tag_representation_dim,
                                                    tag_representation_dim,
                                                    label_dim=num_labels)

        self._pos_tag_embedding = pos_tag_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        # add a head sentinel to accommodate for extra root token
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(representation_dim, encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(tag_representation_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")
        check_dimensions_match(arc_representation_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")
        
        # the unlabelled_f1 is confirmed the same from both classes
        self._unlabelled_f1 = F1Measure(positive_label=1)
        self._enhanced_attachment_scores = EnhancedAttachmentScores()
        
        self._arc_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._tag_loss = torch.nn.CrossEntropyLoss(reduction='none')
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pos_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                enhanced_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        pos_tags : torch.LongTensor, optional (default = None)
            The output of a ``SequenceLabelField`` containing POS tags.
        metadata : List[Dict[str, Any]], optional (default = None)
            A dictionary of metadata for each batch element which has keys:
                tokens : ``List[str]``, required.
                    The original string tokens in the sentence.
        enhanced_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length, sequence_length)``.

        Returns
        -------
        An output dictionary.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(tokens)

        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        float_mask = mask.float()
        encoded_text = self._dropout(encoded_text)

        batch_size, _, encoding_dim = encoded_text.size()
        #print(batch_size, _, encoding_dim)

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)

        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)

        #mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
        float_mask = mask.float()

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))

        # shape (batch_size, sequence_length, sequence_length)
        arc_scores = self.arc_attention(head_arc_representation,
                                        child_arc_representation)
        
        # shape (batch_size, num_tags, sequence_length, sequence_length)
        arc_tag_logits = self.tag_bilinear(head_tag_representation,
                                           child_tag_representation)

        # Switch to (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1).contiguous()

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        arc_scores = arc_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        arc_probs, arc_tag_probs = self._greedy_decode(arc_scores,
                                                       arc_tag_logits,
                                                       mask)

        output_dict = {
                "arc_probs": arc_probs,
                "arc_tag_probs": arc_tag_probs,
                "mask": mask,
                }

        if metadata:
            output_dict["ids"] = [meta["ids"] for meta in metadata]
            output_dict["tokens"] = [meta["tokens"] for meta in metadata]
            output_dict["head_tags"] = [meta["head_tags"] for meta in metadata]
            output_dict["head_indices"] = [meta["head_indices"] for meta in metadata]
            

        if enhanced_tags is not None:
            arc_nll, tag_nll = self._construct_loss(arc_scores=arc_scores,
                                                    arc_tag_logits=arc_tag_logits,
                                                    enhanced_tags=enhanced_tags,
                                                    mask=mask)
            output_dict["loss"] = arc_nll + tag_nll
            output_dict["arc_loss"] = arc_nll
            output_dict["tag_loss"] = tag_nll

            # Make the arc tags not have negative values anywhere
            # (by default, no edge is indicated with -1).
            arc_indices = (enhanced_tags != -1).float()
            tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
            one_minus_arc_probs = 1 - arc_probs
            # We stack scores here because the f1 measure expects a
            # distribution, rather than a single value.
            self._unlabelled_f1(torch.stack([one_minus_arc_probs, arc_probs], -1), arc_indices, tag_mask)
                
            # get unlabeled and labeled f1  
            output_dict = self.decode(output_dict)
            
            # predicted arcs, arc_tags
            predicted_indices = output_dict["arcs"]
            #print("predicted_indices", predicted_indices)
            predicted_arc_tags = output_dict["arc_tags"]
            #print("predicted_arc_tags", predicted_arc_tags)
            predicted_labeled_arcs = output_dict["labeled_arcs"]
            #print("predicted_labeled_arcs", predicted_labeled_arcs)
            
            # gold arcs, arc_tags
            gold_arcs = [meta["arc_indices"] for meta in metadata]
            #print("gold arcs", gold_arcs)
            gold_arc_tags = [meta["arc_tags"] for meta in metadata]
            #print("gold arc tags", gold_arc_tags)
            gold_labeled_arcs = [meta["labeled_arcs"] for meta in metadata]
            #print("gold labeled arcs", gold_labeled_arcs)
            
            self._enhanced_attachment_scores(predicted_indices, predicted_arc_tags, predicted_labeled_arcs, \
                                             gold_arcs, gold_arc_tags, gold_labeled_arcs, tag_mask)
            
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        arc_tag_probs = output_dict["arc_tag_probs"].cpu().detach().numpy()
        #print("arc_tag_probs", arc_tag_probs)
        arc_probs = output_dict["arc_probs"].cpu().detach().numpy()
        mask = output_dict["mask"]
        lengths = get_lengths_from_binary_sequence_mask(mask)
        arcs = []
        arc_tags = []
        # append arc and label to calculate ELAS
        labeled_arcs = []
        
        for instance_arc_probs, instance_arc_tag_probs, length in zip(arc_probs, arc_tag_probs, lengths):
            arc_matrix = instance_arc_probs > self.edge_prediction_threshold

            edges = []
            edge_tags = []
            edges_and_tags = []
            
            # dictionary where a word has been assigned a head
            found_heads = {}
            
            # set each label to False but will be updated as True if the word has a head over the threshold.
            for i in range(length):
                found_heads[i] = False
            
            # i is whether the word is a head
            for i in range(length):            
                for j in range(length):             
                    # check if an edge exists in the predicted adjacency matrix.
                    if arc_matrix[i, j] == 1:
                        head_modifier_tuple = (i, j)
                        edges.append(head_modifier_tuple)
                        tag = instance_arc_tag_probs[i, j].argmax(-1)
                        edge_tags.append(self.vocab.get_token_from_index(tag, "labels"))
                        # append ((h,m), label) tuple
                        edges_and_tags.append((head_modifier_tuple, self.vocab.get_token_from_index(tag, "labels")))
                        found_heads[j] = True
                        
            # some words won't have found heads so we will find the edge with the highest probability for each unassigned word
            # could lower the threshold for these cases or else just do the max
            head_information = found_heads.items()
            unassigned_tokens = []
            for (word, has_found_head) in head_information:
                # not interested in selecting heads for the dummy ROOT token
                if has_found_head == False and word != 0:
                    unassigned_tokens.append(word)
                                
            if len(unassigned_tokens) >= 1:
                head_choices = {unassigned_token: [] for unassigned_token in unassigned_tokens}
                
                # keep track of the probabilities of the other words being heads of the unassigned tokens
                for i in range(length):
                    for j in unassigned_tokens:
                        # edge
                        head_modifier_tuple = (i, j)
                        # score
                        probability = instance_arc_probs[i, j]
                        head_choices[j].append((head_modifier_tuple, probability))
                        
                for unassigned_token, edge_score_tuples in head_choices.items():
                    # get the best edge for each unassigned token based on the score which is element [1] in the tuple.
                    best_edge = max(edge_score_tuples, key = itemgetter(1))[0]
                    #print("best edge!", best_edge)
                    
                    edges.append(best_edge)
                    tag = instance_arc_tag_probs[best_edge].argmax(-1)                   
                    edge_tags.append(self.vocab.get_token_from_index(tag, "labels"))
                    edges_and_tags.append((best_edge, self.vocab.get_token_from_index(tag, "labels")))

            arcs.append(edges)
            arc_tags.append(edge_tags)
            labeled_arcs.append(edges_and_tags)

        output_dict["arcs"] = arcs  
        output_dict["arc_tags"] = arc_tags
        output_dict["labeled_arcs"] = labeled_arcs
        
        return output_dict

    def _construct_loss(self,
                        arc_scores: torch.Tensor,
                        arc_tag_logits: torch.Tensor,
                        enhanced_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for an adjacency matrix.

        Parameters
        ----------
        arc_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate a
            binary classification decision for whether an edge is present between two words.
        arc_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to generate
            a distribution over edge tags for a given edge.
        enhanced_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The labels for every arc.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        arc_indices = (enhanced_tags != -1).float()
        # Make the arc tags not have negative values anywhere
        # (by default, no edge is indicated with -1).
        enhanced_tags = enhanced_tags * arc_indices
        arc_nll = self._arc_loss(arc_scores, arc_indices) * float_mask.unsqueeze(1) * float_mask.unsqueeze(2)
        # We want the mask for the tags to only include the unmasked words
        # and we only care about the loss with respect to the gold arcs.
        tag_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(2) * arc_indices

        batch_size, sequence_length, _, num_tags = arc_tag_logits.size()
        original_shape = [batch_size, sequence_length, sequence_length]
        reshaped_logits = arc_tag_logits.view(-1, num_tags)
        reshaped_tags = enhanced_tags.view(-1)
        tag_nll = self._tag_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask

        valid_positions = tag_mask.sum()

        arc_nll = arc_nll.sum() / valid_positions.float()
        tag_nll = tag_nll.sum() / valid_positions.float()
        return arc_nll, tag_nll

    @staticmethod
    def _greedy_decode(arc_scores: torch.Tensor,
                       arc_tag_logits: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently.

        Parameters
        ----------
        arc_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        arc_tag_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to
            generate a distribution over tags for each arc.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length).

        Returns
        -------
        arc_probs : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, sequence_length) representing the
            probability of an arc being present for this edge.
        arc_tag_probs : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags)
            representing the distribution over edge tags for a given edge.
        """
        # Mask the diagonal, because we don't self edges.
        inf_diagonal_mask = torch.diag(arc_scores.new(mask.size(1)).fill_(-numpy.inf))
        arc_scores = arc_scores + inf_diagonal_mask
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits + inf_diagonal_mask.unsqueeze(0).unsqueeze(-1)
        # Mask padded tokens, because we only want to consider actual word -> word edges.
        minus_mask = (1 - mask).to(dtype=torch.bool).unsqueeze(2)
        arc_scores.masked_fill_(minus_mask, -numpy.inf)
        arc_tag_logits.masked_fill_(minus_mask.unsqueeze(-1), -numpy.inf)
        # shape (batch_size, sequence_length, sequence_length)
        arc_probs = arc_scores.sigmoid()
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)
        return arc_probs, arc_tag_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        
        #metrics_to_track = ["EUAS", "ELAS"]
        metrics_to_track = ["unlabeled_precision", "unlabeled_recall", "unlabeled_f1", "labeled_precision", "labeled_recall", "labeled_f1"]
        
        # get tree scores
        tree_results_dict = self._enhanced_attachment_scores.get_metric(reset)
        for metric, value in tree_results_dict.items():
            if metric in metrics_to_track:
                metrics[metric] = value
        
        precision, recall, f1_measure = self._unlabelled_f1.get_metric(reset)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1_measure
        return metrics
