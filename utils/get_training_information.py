"""
- Gets the number of training sentences and warmup steps for UDify.
- Splits training treebanks into groups based on sentence lengths.
  python utils/get_training_information.py output --dataset_dir data/train-dev
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Any, Sequence, Union

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


def get_training_information(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, List[int]]:    
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
            training_info_dict[train_file] = [sentence_count, num_warmup_steps]
    
    return training_info_dict
            

def create_jobs(training_info_dict: Dict[str, List[int]]) -> Dict[str, Sequence[Sequence[Union[str, int]]]]: 

    """
    Splits the treebanks into different groups in a round-robin fashion based on decreasing treebank size.
    :param training_info_dict: a dictionary mapping a training file to the number of training sents and warmup steps.
    :return: a dictionary containing mappings between job groups and tbid, sentence length pairs.
    """
    # sort treebank metadata by decreasing size
    sorted_metadata = sorted(training_info_dict.items(), key=lambda key: key[1], reverse = True)
    
    # assign treebanks to job lists in round-robin fashion    
    jobs_list = ["jobs_1", "jobs_2", "jobs_3", "jobs_4", "jobs_5"]
    jobs_dict = {job: [] for job in jobs_list}

    i = 1
    num_job_buckets = len(jobs_list)
    for treebank, (num_sents, num_warmup_steps) in sorted_metadata:
        tbid = treebank.split("-")[0]
        job = "jobs_" + str(i)
        jobs_dict[job].append([tbid, num_sents])
        i += 1
        if i > num_job_buckets:
            # start over
            i = 1     
    
    return jobs_dict
    

if __name__ == '__main__':
    training_info_dict = get_training_information(args.dataset_dir)
    print("")
    print("training files with sentence lengths and numbers of warmup steps: \n")
    print(training_info_dict)
    
    jobs_dict = create_jobs(training_info_dict)
    print("")
    print("job groups based on training size: \n")
    print(jobs_dict)
    print("")
    
    meta_out = os.path.join(args.output_dir, "training_information.json")
    with open(meta_out, "w") as f:
        json.dump(training_info_dict, f, indent=2)    

    jobs_out = os.path.join(args.output_dir, "training_jobs.json")
    with open(jobs_out, "w") as f:
        json.dump(jobs_dict, f, indent=2)
        