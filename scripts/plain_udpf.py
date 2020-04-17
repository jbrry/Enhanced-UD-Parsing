#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import os
import subprocess
import sys

import common_udpipe_future
import utilities

def supports_lcode(lcode):
    # This parser supports any language for which the user
    # has training data.
    return True

def can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
    return True

def has_ud25_model_for_tbid(tbid):
    return False

def has_task_model_for_tbid(tbid):
    return False

def train_model_if_missing(lcode, init_seed, datasets, options):
    details = utilities.get_training_details(
        lcode, init_seed, datasets, options, 'plain_udpf',
    )
    tr_data_filename, monitoring_datasets, model_dir, epochs = details
    if tr_data_filename is None:
        return None
    return train(
        tr_data_filename, init_seed, model_dir,
        monitoring_datasets = monitoring_datasets,
        epochs = epochs,
        priority = 20,
        is_multi_treebank = '+' in datasets,
        submit_and_return = True,
    )

def train(
    dataset_filename, seed, model_dir,
    epoch_selection_dataset = None,
    monitoring_datasets = [],
    batch_size = 32,
    epochs = 60,
    priority = 50,
    is_multi_treebank = False,
    submit_and_return = False,
):
    if epoch_selection_dataset:
        raise ValueError('Epoch selection not supported with udpipe-future.')
    command = []
    command.append('scripts/plain_udpf-train.sh')
    command.append(dataset_filename)
    if seed is None:
        raise NotImplementedError
    command.append(seed)
    command.append(model_dir)
    command.append('%d' %batch_size)
    command.append(common_udpipe_future.get_training_schedule(epochs))
    if is_multi_treebank:
        command.append('--extra_input tbemb')  # will be split into 2 args in wrapper script
    else:
        command.append('')
    for i in range(2):
        if len(monitoring_datasets) > i:
            conllu_file = monitoring_datasets[i]
            if type(conllu_file) is tuple:
                conllu_file = conllu_file[0]
            if type(conllu_file) is not str:
                conllu_file = conllu_file.filename
            command.append(conllu_file)
    task = common_udpipe_future.run_command(
        command,
        priority = 200+priority,
        submit_and_return = submit_and_return,
    )
    if submit_and_return:
        return task
    check_model(model_dir)

def check_model(model_dir):
    if not os.path.exists(model_dir):
        raise ValueError('Failed to train parser (missing output)')
    if common_udpipe_future.incomplete(model_dir):
        # do not leave erroneous model behind
        if common_udpipe_future.memory_error(model_dir):
            # out-of-memory error detected
            error_name = model_dir + '-oom'
        else:
            # other errors
            error_name = model_dir + '-incomplete'
        os.rename(model_dir, error_name)
        raise ValueError('Model is missing essential files: ' + error_name)

def get_model_id(lcode, init_seed, dataset, options):
    return utilities.get_model_dir(
        'plain_udpf', lcode, init_seed, datasets, options,
    )[1]

def predict(
    lcode, init_seed, datasets, options,
    conllu_input, conllu_output,
):
    is_multi_treebank = '+' in datasets
    model_path = utilities.get_model_dir(
        'plain_udpf', lcode, init_seed, datasets, options,
    )[0]
    if model_path is None:
        raise ValueError('Request to predict with a model for which training is not supported')
    if not os.path.exists(model_path):
        if options.debug:
            print('Model %s not found' %model_path)
        return False
    command = []
    command.append('scripts/plain_udpf-predict.sh')
    command.append(model_path)
    command.append(conllu_input)
    command.append(conllu_output)
    if is_multi_treebank:
        command.append('--extra_input tbemb')  # split into 2 args by wrapper script
    if options.debug:
        print('Running', command)
        sys.stderr.flush()
        sys.stdout.flush()
    subprocess.call(command)
    # TODO: check output more carefully,
    #       e.g. check number of sentences (=number of empty lines)
    return os.path.exists(conllu_output)

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()

