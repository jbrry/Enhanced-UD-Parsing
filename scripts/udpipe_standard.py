#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
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

def supports_lcode(lcode, tbid = None):
    model_path = get_model_path(lcode, tbid)
    if model_path:
        return True
    else:
        return False

def get_model_path(lcode, tbid = None):
    language = utilities.get_language(lcode).lower()
    udpipe_dir = get_udpipe_dir()
    model_dir = '/'.join((udpipe_dir, 'models'))
    for filename in os.listdir(model_dir):
        # example: czech-cac-ud-2.5-191206.udpipe
        if not filename.endswith('-ud-2.5-191206.udpipe'):
            # other versions not allowed in shared task
            pass
        elif filename.split('-')[0] == language:
            if tbid is None \
            or filename.split('-')[1] == tbid.split('_')[1]:
                return '/'.join((model_dir, filename))
    return None

def can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
    return False

def has_ud25_model_for_tbid(tbid):
    lcode = tbid.split('_')[0]
    return supports_lcode(lcode, tbid)

def has_task_model_for_tbid(tbid):
    lcode = tbid.split('_')[0]
    return supports_lcode(lcode, tbid)

def get_udpipe_dir():
    if 'UDPIPE_DIR' in os.environ:
        return os.environ['UDPIPE_DIR']
    return '/'.join((utilities.get_project_dir(), 'udpipe'))

def get_tbid_from_dataset(dataset):
    if '+' in dataset:
        raise ValueError('No off-the-shelf udpipe model for %r' %datasets)
    _, tbid = dataset.split('.', 1)
    return tbid

def train_model_if_missing(lcode, init_seed, dataset, options, max_tr_tokens = 27000111):
    tbid = get_tbid_from_dataset(dataset)
    assert tbid.startswith(lcode)
    if not supports_lcode(lcode, tbid):
        raise ValueError('No off-the-shelf udpipe model for %s' %tbid)
    print('Using off-the-shelf udpipe model for %s, ignoring init_seed' %tbid)
    return None   # no task to wait for

def predict(lcode, init_seed, dataset, options, raw_input_file, conllu_output_file):
    tbid = get_tbid_from_dataset(dataset)
    assert tbid.startswith(lcode)
    model_path = get_model_path(lcode, tbid)
    if not model_path:
        # try without tbid
        model_path = get_model_path(lcode)
        if model_path:
            print('Falling back to segmenting with %s for %s' %(lcode, tbid))
    if not model_path:
        raise ValueError('Missing udpipe standard segmenter model for %s' %lcode)
    # assemble command line to run udpipe
    command = []
    command.append('bin/udpipe')  # TODO: support alternative locations
    command.append('--tokenize')
    command.append('--output=conllu')
    command.append('--outfile=%s' %conllu_output_file)
    command.append(model_path)
    command.append(raw_input_file)
    if options.debug:
        print('Running', command)
    subprocess.call(command)

