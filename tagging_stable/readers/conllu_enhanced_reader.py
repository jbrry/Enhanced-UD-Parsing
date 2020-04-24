# feature extraction/MWT processing is based on the implementation in: https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/universal_dependencies.py

from typing import Dict, Tuple, List, Any, Callable
import logging

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, TextField, SequenceLabelField, MetadataField
from tagging.fields.rooted_adjacency_field import RootedAdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


def process_multiword_and_elided_tokens(annotation):
    """
    Processes CoNLL-U ids for multi-word tokens and elided tokens.
    When a token is a MWT, the id is set to None so the token is not used in the model.
    Elided token ids are returned as tuples by the conllu library and are converted to a number id here.
    """
    
    for i in range(len(annotation)):
        conllu_id = annotation[i]["id"]
        if type(conllu_id) == tuple:
            if "-" in conllu_id:
                conllu_id = str(conllu_id[0]) + "-" + str(conllu_id[2])
                annotation[i]["multi_id"] = conllu_id
                annotation[i]["id"] = None
                annotation[i]["elided_id"] =  None
            elif "." in conllu_id:
                conllu_id = str(conllu_id[0]) + "." + str(conllu_id[2])
                conllu_id = float(conllu_id)
                annotation[i]["elided_id"] = conllu_id
                annotation[i]["id"] = conllu_id 
                annotation[i]["multi_id"] = None
        else:
            annotation[i]["elided_id"] =  None
            annotation[i]["multi_id"] = None
    
    return annotation


@DatasetReader.register("universal_dependencies_enhanced")
class UniversalDependenciesEnhancedDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.
    # Parameters
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the tokens TextField.
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


    def _convert_deps_to_nested_sequences(self, deps):
        """
        Converts a series of deps labels into relation-lists and head-lists respectively.

        # Parameters
        deps : ``List[List[Tuple[str, int]]]``
            The enhanced dependency relations.
        
        # Returns
        List-of-lists containing the enhanced tags and heads.
        """
        rels = []
        heads = []
            
        for target_output in deps:        
            # check if there is just 1 head
            if len(target_output) == 1:
                rel = [x[0] for x in target_output]
                head = [x[1] for x in target_output]
                rels.append(rel)
                heads.append(head)
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
                
        return rels, heads


    def _process_elided_tokens(self, ids, heads):
        """
        Changes elided token format from tuples to float values.
        We create a dictionary which maps the original CoNLL-U indices to 
        indices based on the order they appear in the sentence.
        This means that when an elided token is encountered, e.g. "8.1",
        we map the index to "9" and offset every other index following this token by +1.
        This process is done every time an elided token is encountered.
        At decoding the time, the (head:dependent) tuples are converted back to the original indices.
        
        # Parameters
        ids : ``List[Union[int, tuple]``
            The original CoNLLU indices of the tokens in the sentence. They will be
            used as keys in a dictionary to map from original to new indices.
        """
        processed_heads = []
        # store the indices of words as they appear in the sentence        
        original_to_new_indices = {}
        # set a placeholder for ROOT
        original_to_new_indices[0] = 0
        
        for token_index, head_list in enumerate(heads):
            conllu_id = ids[token_index]
            # map the original CoNLL-U IDs to the new 1-indexed IDs
            original_to_new_indices[conllu_id] = token_index + 1
            current_heads = []
            for head in head_list:
                # convert copy node tuples: (8, '.', 1) to float: 8.1
                if type(head) == tuple:
                    # join the values in the tuple
                    copy_node = str(head[0]) + '.' + str(head[2])
                    copy_node = float(copy_node)
                    current_heads.append(copy_node)
                else:
                    # regular head id
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

        return original_to_new_indices, augmented_heads


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)
            
            for annotation in parse_incr(conllu_file):
                conllu_metadata = []
                metadata = annotation.metadata
                for k, v in metadata.items():
                    metadata_line = (f"# {k} = {v}")
                    conllu_metadata.append(metadata_line)
    
                self.contains_elided_token = False
                annotation = process_multiword_and_elided_tokens(annotation)
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
                elided_tokens = [x for x in annotation if x["elided_id"] is not None]
                if len(elided_tokens) >= 1:
                    self.contains_elided_token = True
                
                # considers all tokens except MWTs for prediction
                annotation = [x for x in annotation if x["id"] is not None]
                
                if len(annotation) == 0:
                    continue

                def get_field(tag: str, map_fn: Callable[[Any], Any] = None) -> List[Any]:
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]
                
                # Extract multiword token rows (not used for prediction, purely for evaluation)
                ids = [x["id"] for x in annotation]
                multiword_ids = [x["multi_id"] for x in multiword_tokens]
                multiword_forms = [x["form"] for x in multiword_tokens]
                tokens = get_field("form")
                lemmas = get_field("lemma")
                upos_tags = get_field("upostag")
                xpos_tags = get_field("xpostag")
                feats = get_field("feats", lambda x: "|".join(k + "=" + v for k, v in x.items())
                                     if hasattr(x, "items") else "_")
                
                misc = get_field("misc", lambda x: "|".join(k + "=" + v if v is not None else "" for k, v in x.items())
                                    if hasattr(x, "items") else "_")
                                
                heads = get_field("head")
                dep_rels = get_field("deprel")
                dependencies = list(zip(dep_rels, heads))
                deps = get_field("deps")
                   
                yield self.text_to_instance(tokens, lemmas, upos_tags, xpos_tags,
                                            feats, dependencies, deps, ids, misc,
                                            multiword_ids, multiword_forms, conllu_metadata)
     
    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        lemmas: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        feats: List[str] = None,
        dependencies: List[Tuple[str, int]] = None,
        deps: List[List[Tuple[str, int]]] = None,
        ids: List[str] = None,
        misc: List[str] = None,
        multiword_ids: List[str] = None,
        multiword_forms: List[str] = None,
        conllu_metadata: List[str] = None,
        contains_elided_token: bool = False,
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

        # NOTE: in order for this reader to work with the old models, we need to remove some
        # of the features as the old model only uses upos, and we need to change some names.
        # we can release this version so that experiments are reproducible but then add back 
        # in the full functionality to the stable version so that extra features can be used there.

        token_field  = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        
        fields["pos_tags"] = SequenceLabelField(upos_tags, token_field, label_namespace="pos")

        # new functionality
        #names = ["upos", "xpos", "lemmas"]
        #all_tags = [upos_tags, xpos_tags, lemmas]
        #for name, field in zip(names, all_tags):
        #    if field:
        #        fields[name] = SequenceLabelField(field, token_field, label_namespace=name)        
        
        #sublist_fields = []
        #for atomic_feat in feats:
        #    feat_fields = ListField([LabelField(feat, label_namespace="feats")
        #                              for feat in atomic_feat.split("|")])   
        #    sublist_fields.append(feat_fields)
        #fields["feats"] = ListField(sublist_fields)
        
        # basic dependency tree
        if dependencies is not None:
            head_tags = [x[0] for x in dependencies]
            head_indices = [x[1] for x in dependencies]
            # we're not using the basic tree in the parse at the moment
            # so we are excluding these fields.
            #fields["head_tags"] = SequenceLabelField(
            #    [x[0] for x in dependencies], token_field, label_namespace="head_tags"
            #)
            #fields["head_indices"] = SequenceLabelField(
            #    [x[1] for x in dependencies], token_field, label_namespace="head_index_tags"
            #)
        
        # enhanced dependencies
        if deps is not None:
            enhanced_arc_tags, enhanced_arc_indices = self._convert_deps_to_nested_sequences(deps)
            # extra processing is needed if a sentence contains an elided token
            if self.contains_elided_token == True:
                original_to_new_indices, augmented_heads = self._process_elided_tokens(ids, enhanced_arc_indices)
                enhanced_arc_indices = augmented_heads
            else:
                original_to_new_indices = None
     
            assert len(enhanced_arc_tags) == len(enhanced_arc_indices), "each arc should have a label"

            arc_indices = []
            arc_tags = []
            arc_indices_and_tags = []
            
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
                #fields["enhanced_tags"] = RootedAdjacencyField(arc_indices, token_field_with_root, arc_tags, label_namespace="deps")
        
        fields["metadata"] = MetadataField({
            "tokens": tokens,
            "upos_tags": upos_tags,
            "xpos_tags": xpos_tags,
            "feats": feats,
            "lemmas": lemmas,
            "ids": ids,
            "misc": misc,
            "original_to_new_indices": original_to_new_indices,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "arc_indices": arc_indices,
            "arc_tags": arc_tags,
            "labeled_arcs": arc_indices_and_tags,
            "multiword_ids": multiword_ids,
            "multiword_forms": multiword_forms,
            "conllu_metadata": conllu_metadata
        })

        return Instance(fields)
