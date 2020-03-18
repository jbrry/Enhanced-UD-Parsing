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
                return True
    return False

def can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
    return False

def has_ud25_model_for_tbid(tbid):
    lcode = tbid.split('_')[0]
    return supports_lcode(lcode, tbid)

def get_udpipe_dir():
    if 'UDPIPE_DIR' in os.environ:
        return os.environ['UDPIPE_DIR']
    return '/'.join((utilities.get_project_dir(), 'udpipe'))

