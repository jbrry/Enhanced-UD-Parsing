from typing import Dict, List, Iterator, Union, Set, Sequence, Optional, cast
import logging

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length

logger = logging.getLogger(__name__)


class SequenceMultiLabelField(Field[torch.Tensor]):
    """
    A `SequenceMultiLabelField` is an extension of the :class:`MultiLabelField` that allows for sequences of multiple labels.
    It is particularly useful in sequential multi-label classification where each token may have numerous correct labels.
    As with the :class:`LabelField`, labels are either strings of text or 0-indexed integers (if you wish
    to skip indexing by passing skip_indexing=True).
        If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.
    
    # Parameters    
    labels : `Sequence[Sequence[Union[str, int]]]`
    label_namespace : `str`, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the `Vocabulary` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    skip_indexing : `bool`, optional (default=False)
        If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
        step.  If this is `False` and your labels are not strings, this throws a `ConfigurationError`.
    num_labels : `int`, optional (default=None)
        If `skip_indexing=True`, the total number of possible labels should be provided, which is required
        to decide the size of the output tensor. `num_labels` should equal largest label id + 1.
        If `skip_indexing=False`, `num_labels` is not required.
    """
    """
    TODO ? This field will get converted into a vector of length equal to the vocabulary size with
    one hot encoding for the labels (all zeros, and ones for the labels).
    # Parameters
    labels : `Sequence[Union[str, int]]`
    """
    
    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()
    
    def __init__(
        self,
        labels: Sequence[Sequence[Union[str, int]]],
        #labels: List[List[Union[str, int]]],
        #sequence_field: SequenceField,
        label_namespace: str = "labels",
        skip_indexing: bool = False,
        num_labels: Optional[int] = None,
    ) -> None:
        self.labels = labels
        self._label_namespace = label_namespace
        self._label_ids = None
        self._maybe_warn_for_namespace(label_namespace)
        self._num_labels = num_labels

        
#        if skip_indexing and self.labels:
        
#            # first check if its a list
#            if not all(isinstance(label, list) for label in labels):
#                raise ConfigurationError(
#                    "SequenceMultiLabelFields expects lists if skip_indexing=False. "
#                    "Found labels: {}".format(labels)   
#                )
#            
        
#            # then check each item in the list is an int...
#            for li in labels:
#                if not all(isinstance(label, int) for label in li):
#                    raise ConfigurationError(
#                        "In order to skip indexing, your labels must be integers. "
#                        "Found labels = {}".format(labels)
#                )
#            if not num_labels:
#                raise ConfigurationError("In order to skip indexing, num_labels can't be None.")
#
#            if not all(cast(int, label) < num_labels for label in labels):
#                raise ConfigurationError(
#                    "All labels should be < num_labels. "
#                    "Found num_labels = {} and labels = {} ".format(num_labels, labels)
#                )
#
#            self._label_ids = labels
#        else:
            # first check if its a list
            
            
        if not all(isinstance(label, list) for label in labels):
            raise ConfigurationError(
                "SequenceMultiLabelFields expects a list of candidate labels for each input. "
                "Found labels: {}".format(labels)   
            )
        
        # first check if the labels are integers
        self._skip_indexing = False
        for li in labels:
            if all([isinstance(label, int) for label in li]):
                self._label_ids = labels
                self._skip_indexing = True
            
        if self._skip_indexing == False:
            # then check each item in the list is a string
            for li in labels:
                if not all(isinstance(label, str) for label in li):
                    raise ConfigurationError(
                        "SequenceMultiLabelFields expects string labels if skip_indexing=False."
                        "Found labels: {}".format(labels)
                )

                
    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (label_namespace.endswith("labels") or label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning(
                    "Your label namespace was '%s'. We recommend you use a namespace "
                    "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                    "default to your vocabulary.  See documentation for "
                    "`non_padded_namespaces` parameter in Vocabulary.",
                    self._label_namespace,
                )
                self._already_warned_namespaces.add(label_namespace)
                
    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._label_ids is None:
            for li in self.labels:
                for label in li:
                    counter[self._label_namespace][label] += 1  # type: ignore
               
    @overrides
    def index(self, vocab: Vocabulary):
        if self._label_ids is None:
            for li in self.labels:
                # TODO: need to be careful I am not overwriting self._label_ids
                self._label_ids = [vocab.get_token_index(label, self._label_namespace) # type: ignore
                    for label in li]
        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)
            
    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}
        # SLF:
        # return {"num_tokens": self.sequence_field.sequence_length()}
    
    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # quite different in SLF (see there for diffs)
        tensor = torch.zeros(self._num_labels, dtype=torch.long)  # vector of zeros
        if self._label_ids:
            tensor.scatter_(0, torch.LongTensor(self._label_ids), 1)

        return tensor
    
    @overrides
    def empty_field(self):
        # quite different in SLF (see there for diffs)
        return SequenceMultiLabelField([], self._label_namespace, skip_indexing=True)

    def __str__(self) -> str:
        return (
            f"SequenceMultiLabelField with labels: {self.labels} in namespace: '{self._label_namespace}'.'"
        )