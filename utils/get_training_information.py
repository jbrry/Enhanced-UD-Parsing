"""
Gets the number of training sentences and warmup steps for UDify.

    python utils/get_training_information.py output --dataset_dir data/train-dev
"""

import os
import json
import logging
import argparse
from typing import List, Tuple, Dict, Any

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=str, help="The path to output the information.")
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

def get_training_information(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, int]:
    """
    Retrieves the number of sentences and warmup steps (for UDify) per training treebank.
    The values should be the number of steps, e.g. the number of training data sentences / batch size
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    :return: a dictionary mapping a treebank name to the number of warmup steps.
    """
    training_info_dict = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]
        train_file = [file for file in conllu_files if file.endswith("train.conllu")]

        if len(train_file) == 1:
            train_file = train_file.pop()
            train_file_path = os.path.join(dataset_dir, treebank, train_file)            
            
            print("Reading sentences from {}".format(train_file))
            
            # calculate number of sentences
            f = open(train_file_path, 'r', encoding="utf-8") 
            sentence_count = 0
            for line in f.readlines():
                if line.isspace():
                    sentence_count += 1
    
            num_warmup_steps = round(sentence_count / args.batch_size)
            training_info_dict[train_file] = sentence_count, num_warmup_steps

            output_file = os.path.join(args.output_dir, "training_information.json")

            with open(output_file, "w") as f:
                json.dump(training_info_dict, f, indent=2)

if __name__ == '__main__':
    get_training_information(args.dataset_dir)