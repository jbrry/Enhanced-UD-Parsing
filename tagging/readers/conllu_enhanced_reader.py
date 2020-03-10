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
        tokenizer: Tokenizer = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos
        self.tokenizer = tokenizer


    def _convert_deps_to_nested_sequences(self, ids, deps):
        """
        Converts a series of deps labels into head lists and rels lists respectively.
        Processes copy nodes to float values.
        
        If the sentence contains ellided tokens, we create a dictionary which maps the original CoNLLU indices to 
        indices as they appear in the sentence. This means that when an ellided token is encountered, e.g. "8.1",
        we map it to index 9 and offset every other index following this token by +1.
        This process is done every time an ellided token is encountered.
        At decoding the time, the (head : dependent) tuples are converted back to the original indices.
              
        # Parameters
        ids : ``List[Union[int, tuple]``
            The original CoNLLU indices of the tokens in the sentence. They will be
            used as keys in a dictionary to map from original to new indices.
        deps : ``List[List[Tuple[str, int]]]``
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
                # append multiple current target heads/rels together respectively
                current_rels = []
                current_heads = []
                for rel_head_tuple in target_output:
                    current_rels.append(rel_head_tuple[0])
                    current_heads.append(rel_head_tuple[1])
                heads.append(current_heads)
                rels.append(current_rels)
                n_heads.append(len(current_heads))
        
        
        if self.contains_copy_node:
            
            processed_heads = []
            
            # store the indices of words as they appear in the sentence        
            original_to_new_indices = {}
            # set a placeholder for ROOT
            original_to_new_indices[0] = 0
            
            for token_index, head_list in enumerate(heads):
                conllu_id = ids[token_index]
                # map the original CoNLLU IDs to the new 1-indexed IDs
                original_to_new_indices[conllu_id] = token_index + 1

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
            
            # change the indices of the heads to reflect the new order
            augmented_heads = []
            for head_list in processed_heads:
                current_heads = []
                for head in head_list:
                    if head in original_to_new_indices.keys():
                        # take the 1-indexed head based on the order of words in the sentence
                        augmented_head = original_to_new_indices[head]
                        current_heads.append(augmented_head)
                augmented_heads.append(current_heads)
            
            heads = augmented_heads
            
        else:
            original_to_new_indices = None

        return rels, heads, original_to_new_indices


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            num_copy_nodes = 0

            for annotation in parse_incr(conllu_file):
                num_conllu_rows = len(annotation)
                
                # store these for now (might need when decoding)
                multiword_ids = []
                
                # filter the conllu sentences for multi-word tokens which don't have any dependency information
                for i in range(num_conllu_rows):
                    conllu_id = annotation[i]["id"]
                    if type(conllu_id) == tuple:
                        if "-" in conllu_id:
                            multiword_ids.append(conllu_id)
                            annotation[i]["id"] = None
                    else: multiword_ids.append(None)
                
                # only include those annotations where the ids aren't None
                annotation = [x for x in annotation if x["id"] is not None]

                ids = [x["id"] for x in annotation]
                #multiword_ids = [x["multi_id"] for x in multiword_tokens]
                
                # regular case: need to filter out MWTs with no head information but keep elided tokens
                self.contains_copy_node = False
                
                # check for presence of copy nodes (at this point, tuples are only elided tokens as MWT have been removed)
                copy_node = [x for x in annotation if not isinstance(x["id"], int)]
                if copy_node:
                    self.contains_copy_node = True

                    
                    # count number of copy nodes in misc column
                    misc = [x["misc"] for x in annotation]
                    for misc_item in misc:
                        if misc_item is not None:
                            vals = list(misc_item.items())
                            for val in vals:
                                if "CopyOf" in val:
                                    num_copy_nodes += 1
                            
                # fix tuple IDs e.g. (8, '.', 1) to 8.1
                if self.contains_copy_node:
                    for index, conllu_id in enumerate(ids):
                        if type(conllu_id) == tuple:
                            copy_node = list(conllu_id)
                            copy_node = str(copy_node[0]) + '.' + str(copy_node[-1])
                            copy_node = float(copy_node)                            
                            ids[index] = copy_node
                                
                tokens = [x["form"] for x in annotation]
                
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]                
                
                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                deps = [x["deps"] for x in annotation]
    
                yield self.text_to_instance(tokens, pos_tags, list(zip(tags, heads)), deps, ids)
                
        logger.info("Found %s copy nodes ", num_copy_nodes)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        pos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
        deps: List[List[Tuple[str, int]]] = None,
        ids: List[str] = None,
        contains_copy_node: bool = False,
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
        
        fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")
        
        #### basic dependency tree
        if dependencies is not None:
            head_tags = [x[0] for x in dependencies]
            head_indices = [x[1] for x in dependencies]
                       
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
#            fields["head_tags"] = SequenceLabelField(
#                [x[0] for x in dependencies], token_field, label_namespace="head_tags"
#            )
#            fields["head_indices"] = SequenceLabelField(
#                [x[1] for x in dependencies], token_field, label_namespace="head_index_tags"
#            )
        

        #### enhanced deps
        if deps is not None:
            #print(tokens)
            #print(deps)
            enhanced_arc_tags, enhanced_arc_indices, original_to_new_indices = self._convert_deps_to_nested_sequences(ids, deps)
     
            assert len(enhanced_arc_tags) == len(enhanced_arc_indices), "each arc should have a label"

            arc_indices = []
            arc_tags = []
            arc_indices_and_tags = []
            
            # NOTE: this currently assumes every token in the sentence has a head, might not be the case for MWT where head is "_" etc. 
            for modifier, head_list in enumerate(enhanced_arc_indices, start=1):
                for head in head_list:
                    arc_indices.append((head, modifier))

            for relation_list in enhanced_arc_tags:
                for relation in relation_list:
                    arc_tags.append(relation)

            assert len(arc_indices) == len(arc_tags), "each arc should have a label"
            
            for arc_index, arc_tag in zip(arc_indices, arc_tags):
                arc_indices_and_tags.append((arc_index, arc_tag))

            if arc_indices is not None and arc_tags is not None:
                token_field_with_root = ['root'] + tokens
                fields["enhanced_tags"] = RootedAdjacencyField(arc_indices, token_field_with_root, arc_tags)
        
        fields["metadata"] = MetadataField({
            "tokens": tokens,
            "pos_tags": pos_tags,
            #"xpos_tags": xpos_tags,
            #"feats": feats,
            #"lemmas": lemmas,
            "ids": ids,
            "original_to_new_indices": original_to_new_indices,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "arc_indices": arc_indices,
            "arc_tags": arc_tags,
            "labeled_arcs": arc_indices_and_tags
        })

        return Instance(fields)
