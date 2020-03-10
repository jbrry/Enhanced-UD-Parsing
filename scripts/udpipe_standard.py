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

def supports(lcode):
    language = utilities.get_language(lcode).lower()
    udpipe_dir = get_udpipe_dir()
    model_dir = '/'.join((udpipe_dir, 'models'))
    for filename in os.listdir(model_dir):
        if filename.startswith(language):
            return True
    return False

def get_udpipe_dir():
    if 'UDPIPE_DIR' in os.environ:
        return os.environ['UDPIPE_DIR']
    return '/'.join((utilities.get_project_dir(), 'udpipe'))

