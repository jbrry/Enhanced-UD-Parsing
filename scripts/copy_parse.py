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

def supports_lcode(lcode):
    return True

def can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
    return True

def has_ud25_model_for_tbid(tbid):
    return True

def has_task_model_for_tbid(tbid):
    return True

def train_model_if_missing(lcode, init_seed, datasets, options):
    # simple, rule-based system --> nothing to train
    return None

def get_model_id(lcode, init_seed, dataset, options):
    return 'copy2e'

def predict(lcode, init_seed, dataset, options, conllu_input_file, conllu_output_file):
    # copy basic parse to enhanced parse
    f_in = open(conllu_input_file, 'rb')
    f_out = open(conllu_output_file, 'wb')
    while True:
        line = f_in.readline()
        if not line:
            # end of file
            break
        if '\t' in line:
            # token row
            fields = line.split('\t')
            head = fields[6]
            label = fields[7]
            if head != '_' and label != '_':
                fields[8] = head + ':' + label
                line = '\t'.join(fields)
        f_out.write(line)
    f_in.close()
    f_out.close()

