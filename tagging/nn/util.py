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




def mask_tensor(tensor):
    return (tensor != 0).float()

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

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")
        
    # weights mask, not sure if we need these this time..   
    weights = weights.float()
    #print("weights", weights)
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    
    #print("non batch dims", non_batch_dims)
    # shape : (batch_size,)
    # sum weights along the row-dimension
    # correspons to number of active elements for each item in the batch
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    #print("weights batch sum", weights_batch_sum)
    
    
    batch_size = targets.shape[0]    
    seq_len = targets.shape[1]
    nb_labels = targets.shape[-1]

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    #print("== logits flat == \n", logits_flat)
    
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    #print("== log probs flat == \n", log_probs_flat)
    
    # target_flat: (batch * max_len, 1)
    #target_flat = target.view(-1, 1)
    targets_flat = targets.view(-1, nb_labels)
    #print("== targets flat == \n", targets_flat)
    
    # losses_flat: (batch * max_len, 1)
    # dimensions need to be the same size except for the dim you are gathering on
    # gathers from log probs the target index (row dim)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    #print("== losses flat == \n", losses_flat)
    
    # losses: (batch, max_len)
    losses = losses_flat.view(*targets.size())
    #print("== losses == \n", losses)

    # return the tensor where 1s correspond to active elements and 0s are padded elements
    tensor_mask = mask_tensor(targets)
    
    # multiply losses by label mask
    masked_losses = losses * tensor_mask
    #print("== masked losses == \n", masked_losses)
    
    # find the number of 0 elements in losses
    # a 0 here means all elements were activate and n corresponds to how many zeros were in the row
    num_zero_labels = (masked_losses == 0).sum(-1)
    #print("== num zero labels == \n", num_zero_labels)
    
    # possible number of values, multiply by constant of nb_labels
    max_possible_labels = torch.ones(size=(batch_size, seq_len)) * nb_labels
    
    # subtract the number of possible labels by the number of 0-padded elements
    num_active_labels = max_possible_labels - num_zero_labels
    #print("== num active labels == \n", num_active_labels)
  
    # add loss along row-dim then divide by the number of active elements per row
    per_batch_loss = masked_losses.sum(-1) / (num_active_labels + 1e-13)
    #print("== per_batch_loss == \n", per_batch_loss)
    
    # at this point, we are back to 2d shape as we have summed/divided by the number of labels
    # we can use the regular weights values
    # multiply losses by token-level mask
    # not sure if this is necessary as we have already multiplied by mask which should know that some rows
    # are all 0s anyway
    masked_per_batch_loss = per_batch_loss * weights
    #print("== masked_per_batch_loss == \n", masked_per_batch_loss)
    
    
    
    # shape : (batch_size,)
    #print("== masked_per_batch_loss_summmedd == \n", masked_per_batch_loss.sum(non_batch_dims))
    # weights added up
    #print("== weights_batch_sum == \n", weights_batch_sum)
    
    per_batch_loss = masked_per_batch_loss.sum(non_batch_dims) / (weights_batch_sum + 1e-13)

    num_non_empty_sequences = (weights_batch_sum > 0).float().sum() + 1e-13
    #print("== num_non_empty_sequences == \n", num_non_empty_sequences)
    #print(per_batch_loss.sum() / num_non_empty_sequences)
    
    return per_batch_loss.sum() / num_non_empty_sequences