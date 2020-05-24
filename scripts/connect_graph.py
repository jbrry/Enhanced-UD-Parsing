import argparse
from collections import defaultdict
import os
from typing import Dict, List, Tuple
import logging
import codecs
import sys

logger = logging.getLogger(__name__)

FIELDS = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]

parser = argparse.ArgumentParser(description='File utils')
parser.add_argument('--input', '-i', type=str, help='Input CoNLLU file.')
parser.add_argument('--outdir','-o', type=str, help='Directory to write out files to.')
parser.add_argument('--mode', '-m', type=str, default='utf-8', help='The behaviour to connect to fragments: <root_edge>, <best_guess>.')
parser.add_argument('--encoding', '-e', type=str, default='utf-8', help='Type of encoding.')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    logger.info(f"creating outdir in {args.outdir}, files will be written here.")
    os.mkdir(args.outdir)

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

    return nodes_reachable_from_root, keep_searching_for_dependents


def parse_sentence(sentence_blob):
    annotated_sentence = []
    # we don't track comment lines at the moment, but they are completed by conllu-quick-fix.pl
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


def get_lists_of_heads(deps_items):
    """
        get lists of heads for a list of deps items, each item being
        a deps value such as '13:nsubj|19:nsubj:enh|20:nsubj:enh'
    """
    retval = []
    for deps in deps_items:
        if deps == '_' or not deps:
            # won't happen with our parser but let's be robust
            deps = []
        else:
            deps = deps.split("|")
        heads = set()
        for relation in deps:
            head, _ = relation.split(':', 1)
            assert head[0].isdigit()
            heads.add(head)
        retval.append(heads)
    return retval

def without_mwt_ids(inputs):
    """Remove MWT ids."""
    retval = []
    for x in inputs:
        if "-" not in x:
            retval.append(x)
    return retval

def get_reachable(head_to_children, start_id, visited = None):
    if visited is None:
        visited = set()
    if start_id in visited:
        return visited
    visited.add(start_id)
    if not start_id in head_to_children:
        return visited
    for child in head_to_children[start_id]:
        get_reachable(head_to_children, child, visited)
    return visited

def merge_deps(list_of_deps):
    edges = set()
    for deps in list_of_deps:
        if deps == '_' or not deps:
            continue
        for hd in deps.split('|'):
            h, d = hd.split(':', 1)
            edges.add((float(h), d, hd))
    if not edges:
        return '_'
    edges = sorted(list(edges))
    return '|'.join([hd for _, _, hd in edges])

def _read(file_path):
    logger.info("Reading data from: %s", file_path)

    # store each annotated sentence
    conllu_annotations = []

    tree_number = 1
    event_counter = defaultdict(lambda: 0)
    with open(file_path) as conllu_file:
        for annotated_sentence in lazy_parse(conllu_file.read()):
            event_counter['sentence'] += 1
            #print(f"tree number {tree_number}")
            full_ids = [x["id"] for x in annotated_sentence]
            ids = without_mwt_ids(full_ids)

            enhanced_heads = get_lists_of_heads(
                [x["deps"] for x in annotated_sentence if x['id'] in ids]
            )

            # fix; sometimes the parser may not predict a 0:root edge
            has_seen_root = False
            for ehead_list in enhanced_heads:
                if '0' in ehead_list:
                    has_seen_root = True
                    event_counter['has at least one root'] += 1
                    break

            if not has_seen_root:
                event_counter['has no root'] += 1
                # if the enhanced parser didn't predict a root edge, take the root edge
                # from basic and re-try
                for i in range(len(annotated_sentence)):
                    head = annotated_sentence[i]["head"]
                    if head == "0":
                        # fix the annotation
                        annotated_sentence[i]["deps"] = "0:root"   # TODO: not always right to throw away the other edges
                        enhanced_heads[i] = ["0"]
                        event_counter['copied 0:root from basic tree'] += 1
                        break

            assert len(ids) == len(enhanced_heads)

            # dictionary mapping ids to heads
            ids_to_heads = {}
            for conllu_id, head_list in zip(ids, enhanced_heads):
                ids_to_heads[conllu_id] = head_list

            # 1) find roots (UD allows multiple roots in the enhanced graph)
            root_ids = []
            for token_id, heads in ids_to_heads.items():
                if '0' in heads:
                    root_ids.append(token_id)
            assert len(root_ids) > 0

            # 2) find nodes reachable from any of the roots

            # (a) build lookup table head -> children
            head_to_children = {}
            for token_id, heads in ids_to_heads.items():
                for head in heads:
                    if not head in head_to_children:
                        head_to_children[head] = []
                    head_to_children[head].append(token_id)
            # note this should also cover root_ids, so we may not
            # need step 1 after all
            assert sorted(head_to_children['0']) == sorted(root_ids)

            # (b) search for all reachable nodes
            nodes_reachable_from_root = get_reachable(head_to_children, '0')

            #print("reachable nodes", nodes_reachable_from_root)

            # 3) find remaining tokens
            unreachable_nodes = []
            for token_id in ids:
                if token_id not in nodes_reachable_from_root:
                    unreachable_nodes.append(token_id)

            #print("unreachable nodes", unreachable_nodes)

            # for the unreachable nodes we build fragments

            count_unreachable_fragments = 0
            while unreachable_nodes:
                count_unreachable_fragments += 1
                # 4) find an unreachable node that maximises the number of
                #    nodes that can be reached from it but not from root;
                #    this ensures that we do no add a 0:root edge to a node
                #    that has a parent that cannot be reached from the
                #    candidate node and therefore would be a better
                #    candidate, reducing the number of root edges needed
                candidates = []
                best_fragment_size = 0
                for unreachable_node in unreachable_nodes:
                    fragment = get_reachable(
                        head_to_children, unreachable_node
                    )
                    fragment_size = len(fragment)
                    if fragment_size < best_fragment_size:
                        # no good candidate: try next
                        continue
                    elif best_fragment_size < fragment_size:
                        # found better candidate that all before
                        candidates = []
                        best_fragment_size = fragment_size
                    token_seq = float(unreachable_node)
                    candidates.append((token_seq, unreachable_node, fragment))
                # prefer the earliest token all else being equal
                candidates.sort()

                # 5) connect a fragment root to the root-reachable graph
                selected_fragment_root = None
                if args.mode in ('best_guess', 'try_use_basic'):
                    # check wether any of the edges in the basic
                    # tree connects one of the candidate fragments
                    # to the root-reachable fragment
                    for _, fragment_root, fragment in candidates:
                        token_index = full_ids.index(fragment_root)
                        head = annotated_sentence[token_index]['head']
                        if head not in unreachable_nodes:
                            # found a suitable edge
                            selected_fragment_root = fragment_root
                            selected_fragment = fragment
                            label = annotated_sentence[token_index]['deprel']
                            selected_edge = ':'.join((head, label))
                            break
                if selected_fragment_root is None:
                    selected_fragment_root = candidates[0][1]
                    selected_fragment      = candidates[0][2]
                    selected_edge          = '0:root'    # naive solution
                # update graph
                token_index = full_ids.index(selected_fragment_root)
                deps = annotated_sentence[token_index]["deps"]
                deps = merge_deps((deps, selected_edge))
                annotated_sentence[token_index]["deps"] = deps
                head, _ = selected_edge.split(':', 1)
                if not head in head_to_children:
                    head_to_children[head] = []
                head_to_children[head].append(selected_fragment_root)
                # update list of unreachable nodes
                new_unreachable_nodes = []
                for node in unreachable_nodes:
                    if node not in selected_fragment:
                        new_unreachable_nodes.append(node)
                unreachable_nodes = new_unreachable_nodes

            event_counter['connected %3d unreachable fragments' %count_unreachable_fragments] += 1
            conllu_annotations.append(annotated_sentence)
            tree_number += 1

    sys.stderr.write('Statistics:\n')
    total = float(event_counter['sentence'])
    for key in sorted(list(event_counter.keys())):
        value = event_counter[key]
        percentage = 100.0 * value / total
        sys.stderr.write('\t%s\t%d\t%.2f%%\n' %(key, value, percentage))

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

    # metadata
    in_name = os.path.basename(args.input)
    file_string = in_name.split('.')[0]
    tbid = file_string.split('-')[0]
    file_type = file_string.split('-')[-1]

    #out_file_string = (f"{tbid}-ud-{file_type}.conllu")
    out_file_string = in_name
    out_file = os.path.join(args.outdir, out_file_string)

    with codecs.open(out_file, 'w', encoding="utf-8") as f:
        for sentence_blob in decoded_conllu_annotations:
            for sentence in sentence_blob:
                f.write(sentence+'\n')
            f.write('\n')


if __name__ == '__main__':
    # alter the annotation
    conllu_annotations = _read(args.input)

    # prepare output
    decoded_conllu_annotations = decode_conllu_output(conllu_annotations)

    # write output
    conllu_output = write_conllu_output(decoded_conllu_annotations)

