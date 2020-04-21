#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

# For the other allennlp_*_*_*_* model variants, symlink this file.

from __future__ import print_function

import os
import subprocess
import sys

import utilities

def supports_lcode(lcode, tbid = None):
    return True

def can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
    return True

def has_ud25_model_for_tbid(tbid):
    return True

def has_task_model_for_tbid(tbid):
    return True

def train_model_if_missing(lcode, init_seed, datasets, options):
    return None

def get_model_id(lcode, init_seed, dataset, options):
    return utilities.get_model_dir(
        __name__,   # current module's name
        lcode, init_seed, dataset, options,
    )[1]

def predict(
    lcode, init_seed, datasets, options,
    conllu_input, conllu_output,
):
    model_path = utilities.get_model_dir(
        __name__,    # the name of the current module
        lcode, init_seed, datasets, options,
    )[0]
    if model_path is None:
        raise ValueError('Request to predict with a model for which training is not supported')
    if not os.path.exists(model_path):
        if options.debug:
            print('Model %s not found' %model_path)
        return False
    command = []
    command.append('scripts/wrapper-allennlp-enhanced-parser.sh')
    allennlp_version = __name__.split('_')[1]
    command.append(allennlp_version)
    if allennlp_version == '090':
        command.append('tagging_stable')
    elif allennlp_version == 'dev':
        command.append('tagging')
    else:
        raise ValueError('Allennlp version %s not implemented' %allennlp_version)
    # insert dummy enhanced dependencies expected by the parser's conllu reader
    conllu_input_copy2enh = conllu_input + '_c2e'
    copy_parse.predict(
        lcode, init_seed, dataset, options,
        conllu_input, conllu_input_copy2enh
    )
    # compile command to run
    command.append(model_path)
    command.append(conllu_input_copy2enh)
    command.append(conllu_output)
    if options.debug:
        print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)
    # cleanup _c2e file
    os.unlink(conllu_input_copy2enh)
    # TODO: check output more carefully,
    #       e.g. check number of sentences (=number of empty lines)
    return os.path.exists(conllu_output)

