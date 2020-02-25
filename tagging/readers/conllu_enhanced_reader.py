from typing import Dict, Tuple, List
import logging

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, TextField, SequenceLabelField, MetadataField, AdjacencyField
from tagging.fields.sequence_multilabel_field import SequenceMultiLabelField
from tagging.fields.rooted_adjacency_field import RootedAdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


def _convert_deps_to_nested_sequences(deps):
    """
    Converts a series of deps labels into head lists and rels lists respectively.
    Processes copy nodes to float values.
    # Parameters
    deps : `List[List[Tuple[str, int]]]
        The enhanced dependency relations.
    # Returns
    List-of-lists containing the enhanced heads and tags respectively.
    """    
    # We don't want to expand the label namespace with an additional dummy token, so we'll
    # always give the 'ROOT_HEAD' token a label of 'root'.
    
    # create lists to populate target labels for rels and heads.
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
            # append multiple current target heads/rels together respectively.
            current_rels = []
            current_heads = []
            for rel_head_tuple in target_output:
                current_rels.append(rel_head_tuple[0])
                current_heads.append(rel_head_tuple[1])
            heads.append(current_heads)
            rels.append(current_rels)
            n_heads.append(len(current_heads))
            
    # process heads which may contain copy nodes
    processed_heads = []

    for head_list in heads:
        # keep a list-of-lists format.
        current_heads = []
        for head in head_list:
            # convert copy node tuples: (8, '.', 1) to float: 8.1
            if type(head) == tuple:
                copy_node = list(head)
                # join the values in the tuple
                copy_node = str(head[0]) + '.' + str(head[-1])
                #copy_node = float(copy_node)
                copy_node = int(copy_node)
                current_heads.append(copy_node)
            else:
                # regular head index
                current_heads.append(head)
            
        processed_heads.append(current_heads)
                
    return rels, processed_heads


@DatasetReader.register("universal_dependencies_enhanced")
class UniversalDependenciesEnhancedDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.
    # Parameters
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the tokens TextField.
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
                    # count number of copy nodes in misc column.
                    misc = [x["misc"] for x in annotation]
                    for misc_item in misc:
                        if misc_item is not None:
                            vals = list(misc_item.items())
                            for val in vals:
                                if "CopyOf" in val:
                                    num_copy_nodes += 1
                
                ids = [x["id"] for x in annotation]
                # regular case: only uses regular indices.                   
                #annotation = [x for x in annotation if isinstance(x["id"], int)]
                
                tokens = [x["form"] for x in annotation]
                
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]                
                
                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                deps = [x["deps"] for x in annotation]
    
                if self.print_data:
                    print("next example:")
                    print("=" * 15)
                    print("tokens: {} ".format(tokens))
                    print("tags: {} ".format(tags))
                    print("heads: {} ".format(heads))
                    print("deps: {} ".format(deps))
                    print("=" * 15)
                    print("\n")

                yield self.text_to_instance(tokens, pos_tags, list(zip(tags, heads)), deps, ids)
                
        logger.info("Found %s copy nodes ", num_copy_nodes)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        pos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
        deps: List[List[Tuple[str, int]]] = None,
        ids: List[str] = None
    ) -> Instance:

        """
        # Parameters
        tokens : ``List[str]``, required.
            The tokens in the sentence to be encoded.
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
        An instance containing tokens, pos tags, basic and enhanced dependency head tags and head
        indices as fields.
        """

        fields: Dict[str, Field] = {}
        
        token_field  = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field        
        
        fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos_tags")
        
        # basic dependency labels
#        if dependencies is not None:
#            # We don't want to expand the label namespace with an additional dummy token, so we'll
#            # always give the 'ROOT_HEAD' token a label of 'root'.
#            fields["head_tags"] = SequenceLabelField(
#                [x[0] for x in dependencies], token_field, label_namespace="head_tags"
#            )
#            fields["head_indices"] = SequenceLabelField(
#                [x[1] for x in dependencies], token_field, label_namespace="head_index_tags"
#            )


        #### try regular deps
        #if dependencies is not None:
            #arc_indices = []
            #arc_tags = []
            
            #for modifier, target in enumerate(dependencies, start=1):
            #    label = target[0]
            #    head_index = target[1]
            #    print(head_index, "==>", modifier)
            #    arc_indices.append((head_index, modifier))                
            #    arc_tags.append(label)
                
            #if arc_indices is not None and arc_tags is not None:
            #    print(arc_indices)
            #    print(arc_tags)
            #    print("len tokens ", len(tokens))
            #    print("len arc indices ", len(arc_indices))
            #    print("len arc tags ", len(arc_tags))
            #    token_field_with_root = ['root'] + tokens
                #fields["arc_tags"] = RootedAdjacencyField(arc_indices, token_field_with_root, arc_tags)
                #print(fields["arc_tags"])


        #### try regular model with multiple edges (enhanced) (no ROOT)
        if deps is not None:
            enhanced_arc_tags, enhanced_arc_indices = _convert_deps_to_nested_sequences(deps)
     
            assert len(enhanced_arc_tags) == len(enhanced_arc_indices), "each arc should have a label"
            
            # labels need to be 0-indexed for AdjacencyMatrix row-column indexing.
            # which leads to the question, how do we index a ROOT (0) node here?
            # should we just assume that (0, 0) is always ROOT?
            arc_indices = []
            arc_tags = []
            arc_indices_and_tags = []
            
            # model multiple head-dependent relations:
            # for each token, create an edge from the token's head(s) to it, e.g. (h, m)
            # there can be multiple (h, m) tuples for the same modifier.
            # CoNLLU indices start from 1
            # NOTE: this currently assumes every token in the sentence has a head, might not be the case for MWT where head is "_" etc.      
            for modifier, head_list in enumerate(enhanced_arc_indices, start=1):
                for head in head_list:
                    #print(head, "==>", modifier)
                    arc_indices.append((head, modifier))

            for relation_list in enhanced_arc_tags:
                for relation in relation_list:
                    arc_tags.append(relation)

            assert len(arc_indices) == len(arc_tags), "each arc should have a label"
            
            for arc_index, arc_tag in zip(arc_indices, arc_tags):
                arc_indices_and_tags.append((arc_index, arc_tag))

            if arc_indices is not None and arc_tags is not None:
#                print(tokens)
#                print(arc_indices)
#                print(arc_tags)
#                print("len tokens ", len(tokens))
#                print("len arc indices ", len(arc_indices))
#                print("len arc tags ", len(arc_tags))
                
                token_field_with_root = ['root'] + tokens
                fields["enhanced_tags"] = RootedAdjacencyField(arc_indices, token_field_with_root, arc_tags)
                
            fields["metadata"] = MetadataField({"tokens": tokens, "pos_tags": pos_tags, "ids": ids, "arc_indices": arc_indices, "arc_tags": arc_tags, "labeled_arcs": arc_indices_and_tags})

        return Instance(fields)
