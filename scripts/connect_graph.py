from typing import Dict, List, Tuple
import logging
import codecs

logger = logging.getLogger(__name__)

file_path="fr_sequoia_pred.conllu"

FIELDS = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
ids_to_ignore = ["-"]
heads_to_ignore = ["_"]

def traverse_root_children(
    ids_to_heads, 
    nodes_reachable_from_root, 
    keep_searching_for_dependents,
    ):
    """
    :ids_to_heads: dict mapping from conllu id to (enhanced) head.
    :nodes_reachable_from_root: a list of conllu ids (nodes) which are reachable from root.
    :keep_searching_for_dependents: bool flag, whether to keep searching for dependents in the graph.
    """
    num_offspring_before = len(nodes_reachable_from_root)
    
    for token_id, heads in ids_to_heads.items():
        for head in heads:
            if head in nodes_reachable_from_root:
                if token_id not in nodes_reachable_from_root:
                    nodes_reachable_from_root.append(token_id)
    
    num_offspring_after = len(nodes_reachable_from_root)
    # if we didn't add any new children, then we have included all reachable nodes
    if num_offspring_before == num_offspring_after:
        keep_searching_for_dependents = False
        print("done")
    
    return nodes_reachable_from_root, keep_searching_for_dependents


def parse_sentence(sentence_blob):    
    annotated_sentence = []

    lines = [
        line.split("\t")
        for line in sentence_blob.split("\n")
        if line and not line.strip().startswith("#")
    ]
    for line_idx, line in enumerate(lines):
        annotated_token = {k: v for k, v in zip(FIELDS, line)}
        annotated_sentence.append(annotated_token)

    return annotated_sentence


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)


def convert_deps_to_nested_list(deps):
    """
    need to unpack deps items, e.g. '13:nsubj|19:nsubj:enh|20:nsubj:enh'
    """
    enhanced_heads = []
    enhanced_deprels = [] # TODO?
    
    for enhanced_dependency in deps:
        current_heads = []
        current_deprels = []
        split_deps = enhanced_dependency.split("|")             
        # just one edep
        if len(split_deps) == 1:
            split_deps = split_deps.pop()
            head = split_deps.split(":")[0]
            #if head != "_":
            current_heads.append(head)
        # more than one edep
        elif len(split_deps) > 1:
            for split_dep in split_deps:
                head = split_dep.split(":")[0]
                #if head != "_":
                current_heads.append(head)
        
        enhanced_heads.append(current_heads)    
    
    return enhanced_heads


def clean_ids(inputs):
    """Remove MWT ids."""
    for i, x in enumerate(inputs):
        # remove MWT ids
        if "-" in x:
            del inputs[i]
    return inputs

          
def clean_eheads(inputs):
    """Remove null heads."""
    for i, x_list in enumerate(inputs):
        for x in x_list:
            if x == "_":
                del inputs[i]
    return inputs


def _read(file_path):
    logger.info("Reading semantic dependency parsing data from: %s", file_path)
    
    # store each annotated sentence
    conllu_annotations = []
    
    with open(file_path) as sdp_file:
        for annotated_sentence in lazy_parse(sdp_file.read()):

            full_ids = [x["id"] for x in annotated_sentence]
            ids = clean_ids(full_ids)
            
            heads = [x["head"] for x in annotated_sentence]            
            deprels = [x["deprel"] for x in annotated_sentence]
            deps = [x["deps"] for x in annotated_sentence]

            unfiltered_enhanced_heads = convert_deps_to_nested_list(deps)
            enhanced_heads = clean_eheads(unfiltered_enhanced_heads)
            
            assert len(ids) == len(enhanced_heads)
            
            # dictionary mapping ids to heads            
            ids_to_heads = {conllu_id: [] for conllu_id in ids}
            
            for conllu_id, head_list in zip(ids, enhanced_heads):
                for head in head_list:
                    ids_to_heads[conllu_id].append(head)
    
            print("ids 2 heads", ids_to_heads)
                
            # store nodes reachable from root  
            nodes_reachable_from_root = []
            
            # 1) find root
            for token_id, heads in ids_to_heads.items():
                for head in heads:
                    if head == "0":
                        root_index = token_id
                        nodes_reachable_from_root.append(root_index)
            
            # 2) find root's immediate children
            for token_id, heads in ids_to_heads.items():
                for head in heads:
                    if head == root_index:
                        nodes_reachable_from_root.append(token_id)
                    
            keep_searching_for_dependents = True
            while keep_searching_for_dependents:
                nodes_reachable_from_root, keep_searching_for_dependents = traverse_root_children(
                    ids_to_heads, 
                    nodes_reachable_from_root, 
                    keep_searching_for_dependents)
            
            print("reachable nodes", nodes_reachable_from_root)    
            
            # 3) find remaining tokens
            unreachable_nodes = []
            for token_id, heads in ids_to_heads.items():
                    for head in heads:
                        if head != "0" and head not in nodes_reachable_from_root:
                            unreachable_nodes.append(token_id)
            
            print("unreachable nodes", unreachable_nodes)
            
            # for the unreachable nodes, we should try build fragments so we can add
            # an edge to the head of the fragment.
                       
            # 4) find common parents of unreachable tokens          
            unreachable_head_fragments = {}
            
            for unreachable_node in unreachable_nodes:
                unreachable_heads = ids_to_heads[unreachable_node]
                for unreachable_head in unreachable_heads:
                    if unreachable_head not in heads_to_ignore:
                        # set the head of the unreachable nodes as the key and append its children as values
                        unreachable_head_fragments[unreachable_head] = []
            
            for unreachable_node in unreachable_nodes:
                unreachable_heads = ids_to_heads[unreachable_node]
                for unreachable_head in unreachable_heads:
                    if unreachable_head in unreachable_head_fragments:
                        logger.warning(f"found unreachable head {unreachable_head}, creating fragments")
                        # get common children
                        unreachable_head_fragments[unreachable_head].append(unreachable_node)
                    else:
                        raise ValueError(f"could not find head for token {unreachable_node}")
                
            print("unreachable head fragments", unreachable_head_fragments)
            
            # NAIVE SOLUTION: add 0:root to the head of each unreachable fragment
            # n fragments, n labels
            root_edge = "0:root"
            for i in range(len(annotated_sentence)):
                #print(annotated_sentence[i])
                conllu_id = annotated_sentence[i]["id"]
                #print(conllu_id)
                if conllu_id in unreachable_head_fragments.keys():
                    deps_object = annotated_sentence[i]["deps"]
                    altered_deps_object = deps_object + "|" + root_edge
                    annotated_sentence[i]["deps"] = altered_deps_object
                    #print(annotated_sentence[i]["deps"])
            
            conllu_annotations.append(annotated_sentence)
            
    return conllu_annotations


def decode_conllu_output(conllu_annotations):
    decoded_conllu_annotations = []
    for sentence_blob in conllu_annotations:
        conllu_sentence = []
        for conllu_row in sentence_blob:
            lines = [conllu_row[k] for k in FIELDS]
            conllu_lines = "\t".join(lines)
            conllu_sentence.append(conllu_lines)
    
        decoded_conllu_annotations.append(conllu_sentence)
    
    return decoded_conllu_annotations


def write_conllu_output(decoded_conllu_annotations):

    with codecs.open("a.conllu", 'w', encoding="utf-8") as f:
        for sentence_blob in decoded_conllu_annotations:
            for sentence in sentence_blob:
                f.write(sentence+'\n')
            f.write('\n')


if __name__ == '__main__':
    # alter the annotation
    conllu_annotations = _read(file_path)
    
    # prepare output
    decoded_conllu_annotations = decode_conllu_output(conllu_annotations)
    
    # write output
    conllu_output = write_conllu_output(decoded_conllu_annotations)
    