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
    # This parser supports any language for which the user
    # has training data.
    return True

def can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
    return not contains_ud25_data

def has_ud25_model_for_tbid(tbid):
    return False

