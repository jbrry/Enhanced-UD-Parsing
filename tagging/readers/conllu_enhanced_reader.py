from typing import Dict, Tuple, List, cast
import logging

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, TextField, SequenceLabelField, MetadataField, MultiLabelField
from tagging.fields.sequence_multilabel_field import SequenceMultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("universal_dependencies_enhanced")
class UniversalDependenciesEnhancedDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.
    # Parameters
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : ``Tokenizer``, optional, default = None
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        print_data: bool = False,
        tokenizer: Tokenizer = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos
        self.print_data = print_data
        self.tokenizer = tokenizer

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by integers here as elided words have a non-integer word id,
                # as parsed by the conllu python library.
                #annotation = [x for x in annotation if isinstance(x["id"], int)]

                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                
                words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
               
                deps = [x["deps"] for x in annotation]
    
                if self.print_data:
                    #print(words, heads, tags, deps)
                    print("next example:")
                    print("=" * 15)
                    print("words: {}".format(words))
                    print("tags: {}".format(tags))
                    print("heads: {}".format(heads))
                    print("deps: {}:".format(deps))
                    print("=" * 15)
                    print("\n")

                    """
                    next example:
                    ===============
                    words: ['The', 'team', 'who', 'work', 'there', 'are', 'helpfull', ',', 'friendly', 'and', 'extremely', 'knowledgeable', 'and', 'will', 'help', 'you', 'as', 'much', 'as', 'they', 'can', 'with', 'thier', 'years', 'of', 'hands', 'on', 'practice', '.']
                    tags: ['det', 'nsubj', 'nsubj', 'acl:relcl', 'advmod', 'cop', 'root', 'punct', 'conj', 'cc', 'advmod', 'conj', 'cc', 'aux', 'conj', 'obj', 'advmod', 'advmod', 'mark', 'nsubj', 'advcl', 'case', 'nmod:poss', 'obl', 'case', 'compound', 'compound', 'nmod', 'punct']
                    heads: [2, 7, 4, 2, 4, 7, 0, 9, 7, 12, 12, 7, 15, 15, 7, 15, 18, 15, 21, 21, 17, 24, 24, 15, 28, 28, 26, 24, 7]
                    deps: [[('det', 2)], [('nsubj', 4), ('nsubj', 7), ('nsubj', 9), ('nsubj', 12), ('nsubj', 15)], [('ref', 2)], [('acl:relcl', 2)], [('advmod', 4)], [('cop', 7)], [('root', 0)], [('punct', 9)], [('conj:and', 7)], [('cc', 12)], [('advmod', 12)], [('conj:and', 7)], [('cc', 15)], [('aux', 15)], [('conj:and', 7)], [('obj', 15)], [('advmod', 18)], [('advmod', 15)], [('mark', 21)], [('nsubj', 21)], [('advcl:as', 17)], [('case', 24)], [('nmod:poss', 24)], [('obl:with', 15)], [('case', 28)], [('compound', 28)], [('compound', 26)], [('nmod:of', 24)], [('punct', 7)]]


                    list(zip(tags, heads))::
                    [('det', 2), ('nsubj', 7), ('nsubj', 4), ('acl:relcl', 2), ('advmod', 4), ('cop', 7), ('root', 0), ('punct', 9), ('conj', 7), ('cc', 12), ('advmod', 12), ('conj', 7), ('cc', 15), ('aux', 15), ('conj', 7), ('obj', 15), ('advmod', 18), ('advmod', 15), ('mark', 21), ('nsubj', 21), ('advcl', 17), ('case', 24), ('nmod:poss', 24), ('obl', 15), ('case', 28), ('compound', 28), ('compound', 26), ('nmod', 24), ('punct', 7)]

                    deps::
                    [[('det', 2)], [('nsubj', 4), ('nsubj', 7), ('nsubj', 9), ('nsubj', 12), ('nsubj', 15)], [('ref', 2)], [('acl:relcl', 2)], [('advmod', 4)], [('cop', 7)], [('root', 0)], [('punct', 9)], [('conj:and', 7)], [('cc', 12)], [('advmod', 12)], [('conj:and', 7)], [('cc', 15)], [('aux', 15)], [('conj:and', 7)], [('obj', 15)], [('advmod', 18)], [('advmod', 15)], [('mark', 21)], [('nsubj', 21)], [('advcl:as', 17)], [('case', 24)], [('nmod:poss', 24)], [('obl:with', 15)], [('case', 28)], [('compound', 28)], [('compound', 26)], [('nmod:of', 24)], [('punct', 7)]]

                    ===============
                    """

                    # need to parse deps



                yield self.text_to_instance(words, pos_tags, list(zip(tags, heads)), deps)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        upos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
        deps: List[List[Tuple[str, int]]] = None,
    ) -> Instance:

        """
        # Parameters
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies : ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.
        # Returns
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        from itertools import chain

        print(words)
        #print(upos_tags)
        print("DEPENDENCIES")
        print(deps)

        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [Token(t) for t in words]

        text_field = TextField(tokens, self._token_indexers)
        fields["words"] = text_field

        fields["pos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="pos")

        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            
            heads = []
            rels = []

            for target_output in deps:
                # check if there is just 1 head
                if len(target_output) == 1:
                    head = [x[1] for x in target_output]
                    rel = [x[0] for x in target_output]
                    heads.append(head)
                    rels.append(rel)
                # more than 1 head
                else:
                    # append all current target heads/rels to a list
                    current_heads = []
                    current_rels = []
                    for rel_head_tup in target_output:
                        current_heads.append(rel_head_tup[1])
                        current_rels.append(rel_head_tup[0])
                    heads.append(current_heads)
                    rels.append(current_rels)

            print("rels", rels)
            #print("heads", heads)
            #print("input of Multi label field looks like")
            #print([x[0] for x in heads])
            
            #for h in heads:
            #    print("---", h)
            #    fields["head_indices"] = h
            

            #fields["head_indices"] = ListField[ListField[LabelField]]
            
            fields["head_tags"] = SequenceMultiLabelField(rels, label_namespace="head_tags")
            fields["head_indices"] = SequenceMultiLabelField(heads, label_namespace="head_index_tags")
            #ListField([Listfield([LabelField([heads]])])
            
            #ListField([ListField[([x for x in heads]))
            
            
            #ListField[ListField(LabelField([x for x in heads]))]
            
            #ListField([LabelField(x, "logical_form") for x in logical_form[0]])
            
            #MultiLabelField(
            #    [x[0] for x in heads], skip_indexing=True, num_labels = len(heads)+1
            #)

            #fields["head_indices"] = MultiLabelField(
            #    [x for x in heads], text_field, label_namespace="head_index_tags"
            #)








            
            #ListField([
            #        MultiLabelField(label) for label in heads
            #        ], skip_indexing=True)

            #fields["head_indices"] = SequenceLabelField(
            #    [x for x in head_ids], text_field, label_namespace="head_index_tags"
            #)


            fields["metadata"] = MetadataField({"words": words, "pos": upos_tags})
        return Instance(fields)




