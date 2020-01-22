import torch
import torch.nn as nn
from typing import Dict, Optional

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits


@Model.register('dummy_lstm')
class DummyLSTM(Model):
    pass


