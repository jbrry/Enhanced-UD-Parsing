from typing import Dict, Tuple, List, cast
import logging
import collections

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

            num_copy_nodes = 0
            
            for annotation in parse_incr(conllu_file):
                # check for presence of copy nodes
                contains_copy_node = [x for x in annotation if not isinstance(x["id"], int)]
                if contains_copy_node:
                    # count number of copy nodes in misc column
                    misc = [x["misc"] for x in annotation]
                    for misc_item in misc:
                        if misc_item is not None:
                            vals = list(misc_item.items())
                            for val in vals:
                                if "CopyOf" in val:
                                    num_copy_nodes += 1
                
                # regular case: only use regular indices                    
                annotation = [x for x in annotation if isinstance(x["id"], int)]
                
                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                
                words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
               
                deps = [x["deps"] for x in annotation]
    
                if self.print_data:
                    print("next example:")
                    print("=" * 15)
                    print("words: {}".format(words))
                    print("tags: {}".format(tags))
                    print("heads: {}".format(heads))
                    print("deps: {}:".format(deps))
                    print("=" * 15)
                    print("\n")

                yield self.text_to_instance(words, pos_tags, list(zip(tags, heads)), deps)
                
        logger.info("Found %s copy nodes ", num_copy_nodes)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        pos_tags: List[str],
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
        deps : ``List[List[Tuple[str, int]]]``, optional (default = None)
            A list of lists of (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.       
        # Returns
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """

        fields: Dict[str, Field] = {}
        


        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [Token(t) for t in words]

        text_field = TextField(tokens, self._token_indexers)
        fields["words"] = text_field

        fields["pos_tags"] = SequenceLabelField(pos_tags, text_field, label_namespace="pos_tags")
        
        
        # regular
#        if dependencies is not None:
#            # We don't want to expand the label namespace with an additional dummy token, so we'll
#            # always give the 'ROOT_HEAD' token a label of 'root'.
#            fields["normal_head_tags"] = SequenceLabelField(
#                [x[0] for x in dependencies], text_field, label_namespace="head_tags"
#            )
#            fields["normal_head_indices"] = SequenceLabelField(
#                [x[1] for x in dependencies], text_field, label_namespace="head_index_tags"
#            )


        if deps is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            
            # 1) create lists to populate target labels
            rels = []
            heads = []
            n_heads = []
            
            for target_output in deps:
                # check if there is just 1 head
                if len(target_output) == 1:
                    rel = [x[0] for x in target_output]
                    head = [x[1] for x in target_output]
                    rels.append(rel)
                    heads.append(head)
                    n_heads.append(1)
                # more than 1 head
                else:
                    # append multiple current target heads/rels to their own lists
                    current_rels = []
                    current_heads = []
                    for rel_head_tuple in target_output:
                        current_rels.append(rel_head_tuple[0])
                        current_heads.append(rel_head_tuple[1])
                    heads.append(current_heads)
                    rels.append(current_rels)
                    n_heads.append(len(current_heads))
                    
            # 2) need to process heads which may contain copy nodes
            processed_heads = []

            for head_list in heads:
                # keep a list-of-lists format
                current_heads = []
                for head in head_list:
                    # convert copy node tuples: (8, '.', 1) to float: 8.1
                    if type(head) == tuple:
                        copy_node = list(head)
                        # join the values in the tuple
                        copy_node = str(head[0]) + '.' + str(head[-1])
                        copy_node = float(copy_node)
                        current_heads.append(copy_node)
                    else:
                        # regular head index
                        current_heads.append(head)

                processed_heads.append(current_heads)

            assert len(words) == len(heads) == len(processed_heads) == len(n_heads)


            print("words", words)
            print("pos_tags", pos_tags)
            print("dependencies", dependencies)
            print("deps", deps)
            #print("processed heads", processed_heads)
            print("heads", heads)
            print("n heads", n_heads)

            fields["num_heads"] = SequenceLabelField(n_heads, text_field, label_namespace="head_num_tags")
            
            # head_tags : ListField[ListField[LabelField]]
            sublist_fields = []
            for label_list in rels:
                label_fields = ListField([LabelField(label, label_namespace="head_tags")
                                      for label in label_list])   
                sublist_fields.append(label_fields)
            fields["head_tags"] = ListField(sublist_fields)
            
            # head_indices : ListField[ListField[LabelField]]
            sublist_fields = []
            for label_list in processed_heads:
                label_fields = ListField([LabelField(label, label_namespace="head_index_tags", skip_indexing=True)
                                      for label in label_list])   
                sublist_fields.append(label_fields)
            fields["head_indices"] = ListField(sublist_fields)
            

        
            #fields["head_tags"] = SequenceMultiLabelField(rels, text_field, label_namespace="head_tags")
            #fields["head_indices"] = SequenceMultiLabelField(heads, text_field, label_namespace="head_index_tags")

            fields["metadata"] = MetadataField({"words": words, "pos": pos_tags})
        return Instance(fields)




