from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, SequenceLabelField

import itertools
from overrides import overrides
from typing import Dict, List, Iterator


@DatasetReader.register("conll_03_reader")
class CoNLL03DatasetReader(DatasetReader):
    def __init__(self,
                token_indexers: Dict[str, TokenIndexer] = None,
                lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        is_divider = lambda line:line.strip() == ''
        with open(file_path, 'r') as conll_file:
            for divider, lines in itertools.groupby(conll_file, is_divider):
                if not divider:
                    fields = [l.strip().split() for l in lines]
                    # switch it so that each field is a list of tokens/labels
                    fields = [l for l in zip(*fields)]
                    # just take tokens and NER tags
                    tokens, _, _, ner_tags = fields

                    yield self.text_to_instance(tokens, ner_tags)

    @overrides
    def text_to_instance(self,
                        words: List[str],
                        ner_tags: List[str]) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a Token object
        tokens = TextField([Token(w) for w in words], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["label"] = SequenceLabelField(ner_tags, tokens)

        return Instance(fields)


