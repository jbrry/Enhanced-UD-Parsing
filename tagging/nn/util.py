"""
Assorted utilities for working with neural networks in AllenNLP.
"""

import copy
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import math
import numpy
import torch
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


def nested_sequence_cross_entropy_with_logits(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: torch.FloatTensor,
    average: str = "batch",
    label_smoothing: float = None,
    gamma: float = None,
    alpha: Union[float, List[float], torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a nested sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    # Parameters
    logits : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : `torch.LongTensor`, required.
        A `torch.LongTensor` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If `None`, return a vector
        of losses per batch element.
    label_smoothing : `float`, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like `[0.05, 0.05, 0.85, 0.05]` if the 3rd class was
        the correct label.
    gamma : `float`, optional (default = None)
        Focal loss[*] focusing parameter `gamma` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        `gamma` is, the more focus on hard examples.
    alpha : `float` or `List[float]`, optional (default = None)
        Focal loss[*] weighting factor `alpha` to balance between classes. Can be
        used independently with `gamma`. If a single `float` is provided, it
        is assumed binary case using `alpha` and `1 - alpha` for positive and
        negative respectively. If a list of `float` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.
    # Returns
    A torch.FloatTensor representing the cross entropy loss.
    If `average=="batch"` or `average=="token"`, the returned loss is a scalar.
    If `average is None`, the returned loss is a vector of shape (batch_size,).
    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")

    print("targets shape", targets.shape)
    # make sure weights are float
    # weights = (batch_size, sequence_length)
    # corresponds to 1s for non-padded elements and 0 otherwise.
    weights = weights.float()
    #print("weights", weights)
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    #print("non batch dims", non_batch_dims)
    # shape : (batch_size,)
    # sum weights along the row-dimension
    # correspons to number of active elements for each instance in the batch
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    #print("weights_batch_sum", weights_batch_sum)
    # shape : (batch * sequence_length, num_classes)
    # basically merges the labels, getting rid of the batch size dim
    logits_flat = logits.view(-1, logits.size(-1))
    #logits_flat_shape = logits_flat.shape
    print("logit flats shape", logits_flat.shape)
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    print("log_probs_flatshape", log_probs_flat.shape)
    # NB need to flatten again
    # this is because we have n labels per instance
    log_probs_flat_flat = log_probs_flat.view(-1, 1)
    print("log_probs_flat_flat shape", log_probs_flat_flat.shape)
    #print("log probs", log_probs_flat.shape)
    # shape : (batch * max_len, 1)
    
    # make dummy targets 
    
    #targets = torch.LongTensor(32,22,62).random_(0, 10)
    targets_flat = targets.view(-1, 1).long()
    # ([736, 1]) -> ([2112, 1]) because 3 labels per item
    #print("targets flat", targets_flat)
    #print("targets flat shape", targets_flat.shape)
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1. - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):
            # pylint: disable=not-callable
            # shape : (2,)
            alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
                                        dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
            # pylint: disable=not-callable
            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                             '{} provided.').format(type(alpha)))
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        # doesn't like the -1s of label field
        negative_log_likelihood_flat = -torch.gather(log_probs_flat_flat, dim=0, index=targets_flat)
        print("negative_log_likelihood_flat shape", negative_log_likelihood_flat.shape)
        print("negative_log_likelihood_flat", negative_log_likelihood_flat)
    # shape : (batch, sequence_length)    
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    print("negative_log_likelihood shape", negative_log_likelihood.shape)
    print("negative_log_likelihood", negative_log_likelihood)
    
    # custom label masking
    ymask = log_probs_flat_flat.data.new(log_probs_flat_flat.size()).zero_() # (B*SL*NL) all zero
    print(ymask.shape)
    
    ymask.scatter_(0, targets_flat, 1)
    print(ymask.shape)
    
    ymask = Variable(ymask)

    # pluck
    logpy = (log_probs_flat_flat * ymask).sum(1) # this hurts in my heart
    
    
    logpy = logpy.view(*targets.size())
    print("LOGPY", logpy)
    
    
    # shape : (batch, sequence_length)
    
    #negative_log_likelihood = negative_log_likelihood * weights

#    if average == "batch":
#        # shape : (batch_size,)
#        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
#        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
#        return per_batch_loss.sum() / num_non_empty_sequences
#    elif average == "token":
#        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
#    else:
#        # shape : (batch_size,)
#        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
#        return per_batch_loss












def get_label_field_mask(text_field_tensors: Dict[str, torch.Tensor],
                        num_wrapping_dims: int = 0) -> torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
    wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
    is given by ``num_wrapping_dims``.

    If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
    If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
    dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
    if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.

    If the input ``text_field_tensors`` contains the "mask" key, this is returned instead of inferring the mask.

    TODO(joelgrus): can we change this?
    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.ByteTensors  makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.V(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    if "mask" in text_field_tensors:
        print("found mask")
        return text_field_tensors["mask"]

    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    print(tensor_dims)
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))
