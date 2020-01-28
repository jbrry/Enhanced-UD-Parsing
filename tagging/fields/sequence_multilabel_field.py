from typing import Dict, Union, Set, Sequence, Optional
import logging

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import pad_sequence_to_length
from allennlp.common.checks import ConfigurationError


logger = logging.getLogger(__name__)


class SequenceMultiLabelField(Field[torch.Tensor]):
    """
    A `SequenceMultiLabelField` is an extension of the :class:`MultiLabelField` that allows for sequences of multiple labels.
    It is particularly useful in sequential multi-label classification where each input token may have numerous correct labels.
    As with the :class:`LabelField`, labels are either strings of text or 0-indexed integers (if you wish
    to skip indexing by passing skip_indexing=True).
    If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.
    
    This field will get converted into a padded 2-D tensor of [dim_1, dim_2] where dim_1 corresponds to the number of
    items in the input sequence and dim_2 is the value returned by "num_tokens", i.e. the max padding length.
    
    # Parameters    
    labels : `Sequence[Sequence[Union[str, int]]]`
        The input labels to this field which is supposed to be a list-of-lists where the nested list
        either contains strings or integers.
    sequence_field : `SequenceField`
        A field containing the sequence that this `SequenceLabelField` is labeling.  Most often, this is a
        `TextField`, for tagging individual tokens in a sentence.
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
    
    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()
    
    def __init__(
        self,
        labels: Sequence[Sequence[Union[str, int]]],
        sequence_field: SequenceField,
        label_namespace: str = "labels",
        skip_indexing: bool = False,
        num_labels: Optional[int] = None,
    ) -> None:
        self.labels = labels
        self._sequence_field = sequence_field
        self._label_namespace = label_namespace
        self._label_ids = None
        self._maybe_warn_for_namespace(label_namespace)
        self._num_labels = num_labels
        if len(labels) != sequence_field.sequence_length():
            raise ConfigurationError(
                "Label length and sequence length "
                "don't match: %d and %d" % (len(labels), sequence_field.sequence_length())
            )
                
        if not all(isinstance(label_list, list) for label_list in labels):
            raise ConfigurationError(
                "SequenceMultiLabelFields expects a list-of-lists where each sublist contains the labels. "
                "Found labels: {}".format(labels)   
            )
        
        self._skip_indexing = False
        for label_list in labels:
            if all([isinstance(label, int) for label in label_list]):
                self._label_ids = labels
                self._skip_indexing = True
            
        if self._skip_indexing == False:
            for label_list in labels:
                if not all(isinstance(label, str) for label in label_list):
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
            for label_list in self.labels:
                for label in label_list:
                    counter[self._label_namespace][label] += 1  # type: ignore
               
    @overrides
    def index(self, vocab: Vocabulary):
        if self._label_ids is None:
            self._label_ids = []
            for label_list in self.labels:
                current_labels = [vocab.get_token_index(label, self._label_namespace) # type: ignore
                    for label in label_list]
                self._label_ids.append(current_labels)
        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)
    
    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        if self._label_ids:
            dims = []
            for label_list in self.labels:
                label_length = len(label_list)
                dims.append(label_length)
            max_sequence_length = max(dims)          
            return {"num_tokens": max_sequence_length}
        else:
            raise ValueError('Could not find label ids to get padding lengths.')

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        if self._label_ids:
            print(self._label_ids)
            tensors = []
            desired_num_tokens = padding_lengths["num_tokens"]
            for label_list in self._label_ids:
                padded_tags = pad_sequence_to_length(label_list, desired_num_tokens)
                tensor = torch.LongTensor(padded_tags)
                tensors.append(tensor)            
            tensor = torch.stack(tensors)
            print(tensor)
            return tensor

    @overrides
    def empty_field(self):
        return SequenceMultiLabelField([], self._label_namespace, skip_indexing=True)

    def __str__(self) -> str:
        return (
            f"SequenceMultiLabelField with labels: {self.labels} in namespace: '{self._label_namespace}'.'"
        )
