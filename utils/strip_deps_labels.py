"""
Strips enhanced labels to label:<TOKEN>, leaves basic labels unchanged.

python utils/strip_deps_labels.py --dataset_dir data/train-dev data/train-dev-stripped
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
                    help="The path to output the stripped treebanks.")
parser.add_argument("--dataset_dir", default="data/ud-treebanks-v2.5", type=str,
                    help="The path containing the UD treebanks")
parser.add_argument("--treebanks", default=[], type=str, nargs="+",
                    help="Specify a list of treebanks to use; leave blank to default to all treebanks available")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

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
                annotation[i]["id"] = None
                annotation[i]["multi_id"] = None
        else:
            annotation[i]["elided_id"] =  None
            annotation[i]["multi_id"] = None
    
    return annotation

def _convert_deps_to_nested_sequences(deps):
    """
    Converts a series of deps labels into relation-lists and head-lists respectively.
    """
    rels = []
    heads = []
        
    for target_output in deps:
        try:
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
                
        # conllu returns NoneType for mwt/elided
        except TypeError:
            rels.append("_")
            heads.append("_")
            
    return rels, heads


def strip_enhanced_labels(filepath):
    """
    Stores the basic labels in a dictionary.
    Checks to see if the enhanced labels are in the basic label set.
    If not, strips off the enhanced label and appends a dummy <TOKEN> label
    indicating that this token needs to be changed back to an enhanced label.    
    """
    # list to store output conllu sentences which will be written to file
    conllu_output = []
    # store the basic (tree) labels
    basic_labels_dict = {}
    # store the enhanced (graph) labels
    enhanced_labels_dict = {}
    # store the new stripped labels (for visualisation)
    stripped_labels_dict = {}
    
    enhanced_label_count = 0 
    print("Reading sentences from {}".format(filepath))
    with open(filepath, "r") as conllu_file:
        for annotation in parse_incr(conllu_file):
            
            #annotation = process_multiword_and_elided_tokens(annotation)
            # TODO check MWTs/elided
            # considers all tokens except MWTs/elided for prediction
            #annotation = [x for x in annotation if x["id"] is not None]
                      
            # access the heads, deprels and enhanced deprels
            heads = [x["head"] for x in annotation]
            tags = [x["deprel"] for x in annotation]
            deps = [x["deps"] for x in annotation]
            
            # store the basic labels
            for basic_label in tags:
                basic_labels_dict[basic_label] = ""
            
            # collect enhanced rels/heads in nested list format
            enhanced_rels, enhanced_heads = _convert_deps_to_nested_sequences(deps)
            assert len(enhanced_rels) == len(enhanced_heads), "each arc should have a label"
            
            # write out labels
            out_labels = []
            
            for enhanced_label_list in enhanced_rels:
                current_labels = []
                for enhanced_label in enhanced_label_list:
                    # new enhanced label
                    if enhanced_label not in basic_labels_dict:
                        if enhanced_label not in enhanced_labels_dict:
                            enhanced_label_count += 1
                            enhanced_labels_dict[enhanced_label] = ""                            
                        
                        # type: list
                        parts = enhanced_label.split(":")             
                        
                        # unique enhanced rel like "ref"
                        if len(parts) == 1:
                            parts = parts.pop()
                            stripped_label = parts
                            current_labels.append(stripped_label)
                            if stripped_label not in stripped_labels_dict:
                                stripped_labels_dict[stripped_label] = ""
                        
                        
                        # enhanced label
                        elif len(parts) >= 2:
                            start_string = parts[0]
                            label_string = start_string
                            
                            temp = []
                            for part in parts[1:]:
                                label_string = label_string + ":" + part
                                
                                if label_string not in basic_labels_dict: 
                                    stripped_label = start_string + ":" + "<TOKEN>"                 
                                    temp.append(stripped_label)
                                else:
                                    logger.warning(f'\nencountered unusual case from candidate set: \t {tags}  \
                                                   \t and enhanced label: {enhanced_label}')
                                    
                                    stripped_label = start_string + ":" + "<TOKEN>"
                                    temp.append(stripped_label)
                            
                            # catch the shortened label
                            current_labels.append(temp[0])
                    # enhanced label is in basic dict (unchanged)
                    else:
                        stripped_label = enhanced_label
                        current_labels.append(stripped_label)
                        if stripped_label not in stripped_labels_dict:
                            stripped_labels_dict[stripped_label] = ""
                    
                out_labels.append(current_labels)
                
            assert len(tags) == len(out_labels), "mismatched original and stripped tags, got lengths {} and {}".format(len(tags), len(out_labels))

            head_tag_tuples = []
            for head_list, tag_list in zip(enhanced_heads, out_labels):
                current_tuples = []
                for i in range(len(tag_list)):
                    tup = (tag_list[i], head_list[i])
                    current_tuples.append(tup)
                head_tag_tuples.append(current_tuples)
                
            # alter the annotation object to contain the stripped labels
            for i in range(len(annotation)):
                old = annotation[i]["deps"]
                annotation[i]["deps"] = head_tag_tuples[i]
                new = annotation[i]["deps"]
                
                #if old != new:
                #    print(old, "==>", new)
                    
            #print(annotation)
            conllu_sent = annotation.serialize()
            conllu_output.append(conllu_sent)


    return basic_labels_dict, enhanced_labels_dict, stripped_labels_dict, out_labels, conllu_output


def write_stripped_file(in_file: str, treebank:str, conllu_output: List[str]):
    """
    :in_file: the train/dev/test file we are writing
    :treebank: treebank name
    :conllu_output: serialized conllu output
    """
    file_string = in_file.split('.')[0]
    tbid = file_string.split('-')[0]
    mode = file_string.split('-')[-1]
    out_file_string = (f"{tbid}-ud-{mode}.conllu")
    out_file = os.path.join(args.output_dir, treebank, out_file_string)
    print("Writing output to {}".format(out_file))
    with codecs.open(out_file, 'w', encoding="utf-8") as f:
        for sent in conllu_output:
            f.write(sent)
    print("done. \n")


def process_files(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, List[int]]:    
    """

    :param dataset_dir: the directory where all treebank directories are stored.
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    """    
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        
        # make the output Treebank directory
        out_treebank_path = os.path.join(args.output_dir, treebank)
        if not os.path.exists(out_treebank_path):
            os.mkdir(out_treebank_path)        

        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]
        
        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        if len(train_file) == 1:
            train_file = train_file.pop()
            train_file_path = os.path.join(dataset_dir, treebank, train_file)            
            basic_labels_dict, enhanced_labels_dict, stripped_labels_dict, unique_labels, conllu_output = strip_enhanced_labels(train_file_path)
            train_out = write_stripped_file(train_file, treebank, conllu_output)

        if len(dev_file) == 1:
            dev_file = dev_file.pop()
            dev_file_path = os.path.join(dataset_dir, treebank, dev_file)            
            basic_labels_dict, enhanced_labels_dict, stripped_labels_dict, unique_labels, conllu_output = strip_enhanced_labels(dev_file_path)
            dev_out = write_stripped_file(dev_file, treebank, conllu_output)

        if len(test_file) == 1:
            test_file = test_file.pop()
            test_file_path = os.path.join(dataset_dir, treebank, test_file)            
            basic_labels_dict, enhanced_labels_dict, stripped_labels_dict, unique_labels, conllu_output = strip_enhanced_labels(test_file_path)
            test_out = write_stripped_file(test_file, treebank, conllu_output)            
            
if __name__ == '__main__':
    process_files(args.dataset_dir, args.treebanks)
