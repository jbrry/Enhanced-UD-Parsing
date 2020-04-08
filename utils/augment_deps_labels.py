"""
Convert deps labels into basic_label:enhanced_string
"""

from typing import Dict, Tuple, List, Any, Callable
import logging
import os
import json
import argparse
import codecs

from conllu import parse_incr
from conllu.parser import serialize

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=str,
                    help="The path to output the information.")
parser.add_argument("--dataset_dir", default="data/ud-treebanks-v2.5", type=str,
                    help="The path containing all UD treebanks")
parser.add_argument("--config_dir", default="config/ud", type=str,
                    help="The path containing the configuration files")
parser.add_argument("--treebanks", default=[], type=str, nargs="+",
                    help="Specify a list of treebanks to use; leave blank to default to all treebanks available")
parser.add_argument("--batch_size", default=32, type=int,
                    help="The batch size used by the model; the number of training sentences is divided by this number.")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.outdir)


def _convert_deps_to_nested_sequences(deps):
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


def augment_enhanced_labels(filepath):
    """
    Stores the basic labels in a dictionary.
    For all labels that aren't in the basic label vocabulary, adds a new label:<ENHANCED> label.
    """
    
    conllu_output = []
    # store the basic (tree) labels
    basic_labels_dict = {}
    # store the enhanced (graph) labels
    enhanced_labels_dict = {}
    # store the new augmented labels
    augmented_labels_dict = {}
    
    enhanced_label_count = 0 
    print("Reading sentences from {}".format(filepath))
    with open(filepath, "r") as conllu_file:
        for annotation in parse_incr(conllu_file):
                        
            heads = [x["head"] for x in annotation]
            tags = [x["deprel"] for x in annotation]
            deps = [x["deps"] for x in annotation]
            
            # keep track of basic labels
            for basic_label in tags:
                basic_labels_dict[basic_label] = ""
            
            # process enhanced labels
            enhanced_rels, enhanced_heads = _convert_deps_to_nested_sequences(deps)
            assert len(enhanced_rels) == len(enhanced_heads), "each arc should have a label"
            
            # write out labels
            out_labels = []
            for i, enhanced_label_list in enumerate(enhanced_rels):
                current_labels = []
                for enhanced_label in enhanced_label_list:
                    # new enhanced label
                    if enhanced_label not in basic_labels_dict:
                        if enhanced_label not in enhanced_labels_dict:
                            #print("new label: {}".format(enhanced_label))
                            enhanced_label_count += 1
                            enhanced_labels_dict[enhanced_label] = ""                            
                        
                        # type: list
                        parts = enhanced_label.split(":")                      
                        # normal label
                        if len(parts) == 1:
                            parts = parts.pop()
                            augmented_label = parts
                            #print(augmented_label)
                            current_labels.append(augmented_label)
                            if augmented_label not in augmented_labels_dict:
                                augmented_labels_dict[augmented_label] = ""
                        # more than one label or a multi-basic label
                        elif len(parts) == 2:
                            p1 = parts[0]
                            p2 = parts[1]
                            
                            if p1 in basic_labels_dict and p2 not in basic_labels_dict:
                                # we are ok to use the first part which is a basic label
                                augmented_label = p1 + ":" + "<ENHANCED>"
                                #print(augmented_label)
                                current_labels.append(augmented_label)
                                if augmented_label not in augmented_labels_dict:
                                    augmented_labels_dict[augmented_label] = ""
                                    
                            if p1 not in basic_labels_dict and p2 not in basic_labels_dict:
                                raise ValueError("unknown case")
                    # enhanced label is in basic dict
                    else:
                        augmented_label = enhanced_label
                        current_labels.append(augmented_label)
                        if augmented_label not in augmented_labels_dict:
                            augmented_labels_dict[augmented_label] = ""
                    
                out_labels.append(current_labels)
                
            assert len(tags) == len(out_labels), "mismatched head and tags, got lengths {} and {}".format(len(tags), len(out_labels))

            head_tag_tuples = []
            for head_list, tag_list in zip(enhanced_heads, out_labels):
                current_tuples = []
                for i in range(len(head_list)):
                    tup = (tag_list[i], head_list[i])
                    current_tuples.append(tup)
                head_tag_tuples.append(current_tuples)
                
            # alter the annotation to contain the augmented labels
            for i in range(len(annotation)):
                #print(annotation[i]["deps"])
                annotation[i]["deps"] = head_tag_tuples[i]
                #print(annotation[i]["deps"])
                
            annotation_out = annotation.serialize()
            conllu_output.append(annotation_out)

    return basic_labels_dict, enhanced_labels_dict, augmented_labels_dict, out_labels, conllu_output


def process_files(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, List[int]]:    
    """
    Retrieves the number of sentences and warmup steps (for UDify) per training treebank.
    The values should be the number of steps, e.g. the number of training data sentences / batch size
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    :return: a dictionary mapping a treebank name to the number of warmup steps.
    """

    
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]
        
        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        if len(train_file) == 1:
            train_file = train_file.pop()
            train_file_path = os.path.join(dataset_dir, treebank, train_file)
            basic_labels_dict, enhanced_labels_dict, augmented_labels_dict, unique_labels, conllu_output = augment_enhanced_labels(train_file_path)
            
            print("BASIC \n")
            print(len(basic_labels_dict))
            print(basic_labels_dict)
            print("")
            
            print("ENHANCED \n")
            print(len(enhanced_labels_dict))
            print(enhanced_labels_dict)
            print("")
            
            print("AUGMENTED \n")
            print(len(augmented_labels_dict))
            print(augmented_labels_dict)
            print("")
            
            unique = []
            for ek in augmented_labels_dict.keys():
                if ek not in basic_labels_dict:
                    unique.append(ek)
            
            print("UNIQUE \n")    
            print(unique)
            # en_ewt only supported at the moment
            with codecs.open("en_ewt-ud-train-augmented.conllu", 'w', encoding="utf-8") as f:
                for sent in conllu_output:
                    f.write(sent)
 
#        if len(dev_file) == 1:
#            dev_file = dev_file.pop()
#            dev_file_path = os.path.join(dataset_dir, treebank, dev_file)
#            basic_labels_dict, enhanced_labels_dict, augmented_labels_dict, out_labels = augment_enhanced_labels(dev_file_path)       
#        if len(test_file) == 1:
#            test_file = test_file.pop()
#            test_file_path = os.path.join(dataset_dir, treebank, test_file)
#            basic_labels_dict, enhanced_labels_dict, augmented_labels_dict = augment_enhanced_labels(test_file_path)
        
if __name__ == '__main__':
    process_files(args.dataset_dir, args.treebanks)