#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

# TODO: Check that all parts work with Python3 and change
#       shebang above.

from __future__ import print_function

import importlib
import hashlib
import os
import random
import string
import sys
import time

#import utilities

class Options:

    def __init__(self):
        path = sys.argv[0]
        if '/' not in path:
            path = './' + path
        self.scriptdir, self.scriptname = path.rsplit('/', 1)
        need_help = self.read_options()
        if need_help:
            self.print_usage()
            sys.exit()

    def print_usage(self):
        print('Usage: %s [options]' %self.scriptname)
        print("""
Options:

    --final-test            Also make predictions for the test sets of each
                            data set.
                            (Default: only report development set results)

    --workdir  DIR          Path to working directory
                            (Default: . = current directory)

    --taskdir  DIR          Path to shared task data containing a UD-like
                            folder structure with training and dev files.
                            With --final-test, we assume that test files
                            follow the naming and structure of the dev
                            data and can be merged into the train-dev
                            folder provided by the shared task.
                            (Default: workdir + /train-dev)

    --modeldir  DIR         Path to model directory
                            (Default: workdir + /models)

    --predictdir  DIR       Path to prediction directory
                            (Default: workdir + /predictions)

    --init-seed  STRING     Initialise random number generator with STRING.
                            If STRING is empty, use system seed.
                            Will be passed to parser for training with a
                            2-digit suffix. If, for example, the parser can
                            only accepts seeds up to 32767 this seed must
                            not exceed 326 if more than 67 models per
                            treebank are trained or 327 otherwise.
                            (Default: 42)

    --verbose               More detailed log output

    --tolerant              Use existing models and predictions even when
                            the training input or model fingerprint does
                            not match, e.g. because the init seed changed.
                            In case of multiple matching models or
                            predictions, use the newest.
                            (Default: Only re-use models or predictions if
                            the name matches exactly.)

    --dispense  ACTION      When scanning for models or predictions with
                            --tolerant, perform ACTION on all candidates
                            that are not chosen. Actions:
                                none = do nothing
                                log  = write "Dispensable:" and the path
                                rename = add the suffix "-dispensable" to
                                         the name of the file or folder
                                delete = delete the model or folder
                            (Default: none)
""")

    def read_options(self):
        self.final_test = False
        self.workdir    = '.'
        self.taskdir    = None
        self.modeldir   = None
        self.predictdir = None
        self.init_seed  = 42
        self.verbose    = False
        self.debug      = True
        self.tolerant   = False
        self.dispense   = 'none'
        while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
            option = sys.argv[1]
            option = option.replace('_', '-')
            del sys.argv[1]
            if option in ('--help', '-h'):
                return True
            elif option == '--final-test':
                self.final_test = True
            elif option == '--workdir':
                self.workdir = sys.argv[1]
                del sys.argv[1]
            elif option == '--taskdir':
                self.taskdir = sys.argv[1]
                del sys.argv[1]
            elif option == '--modeldir':
                self.modeldir = sys.argv[1]
                del sys.argv[1]
            elif option == '--predictdir':
                self.predictdir = sys.argv[1]
                del sys.argv[1]
            elif option == '--init-seed':
                self.init_seed = sys.argv[1]
                del sys.argv[1]
            elif option == '--verbose':
                self.verbose = True
            elif option == '--debug':
                self.debug = True
            elif option == '--tolerant':
                self.tolerant = True
            elif option == '--dispense':
                self.dispense = sys.argv[1]
                del sys.argv[1]
            else:
                print('Unsupported or not yet implemented option %s' %option)
                return True
        if self.taskdir is None:
            self.taskdir = '/'.join((self.workdir, 'train-dev'))
        if self.modeldir is None:
            self.modeldir = '/'.join((self.workdir, 'models'))
        if self.predictdir is None:
            self.predictdir = '/'.join((self.workdir, 'predictions'))
        if len(sys.argv) != 1:
            return True
        return False

    def get_tasks_and_configs(self):
        self.tasks = []
        self.training_data = []
        self.configs = {}
        for tbname in os.listdir(self.taskdir):
            if not tbname.startswith('UD_'):
                if self.verbose:
                    print('Skipping non-UD entry in taskdir:', tbname)
                continue
            # Scan the treebank folder
            tbdir = '/'.join((self.taskdir, tbname))
            treebank_needs_config = False
            tbid, lcode = None, None
            for filename in os.listdir(tbdir):
                path = '/'.join((tbdir, filename))
                if '-ud-' in filename:
                    tbid, _, suffix = filename.rsplit('-', 2)
                    lcode = tbid.split('_')[0]
                    data_type = suffix.split('.')[0]
                    if '-' in tbid:
                        tbid = tbid.replace('-', '_')
                        if self.verbose:
                            print('Dash in TBID replaced with underscore')
                if filename.endswith('-ud-dev.txt') \
                or filename.endswith('-ud-test.txt'):
                    # found a prediction task
                    treebank_needs_config = True
                    self.tasks.append((path, tbid, lcode, data_type))
                if filename.endswith('-ud-train.conllu'):
                    # found training data
                    self.training_data.append((path, tbid, lcode))
            if not treebank_needs_config:
                continue
            # Find the most specific config available
            # TODO: Add language family as a layer between
            #       lcode and default.
            config_class = None
            for name in [tbid, lcode, 'default']:
                classname = 'Config_' + name
                try:
                    config_class = globals()[classname]
                    break
                except:
                    pass
            if config_class is None:
                raise ValueError('No config found in %r' %(globals().keys()))
            if tbid in self.configs:
                print('Warning: duplicate TBID', tbid)
            self.configs[tbid] = config_class(tbid, lcode)
        if opt_debug:
            print('tasks:', self.tasks)
            print('training_data:', self.training_data)
            print('configs:', self.configs)
    
class Config_default:

    def __init__(self, tbid, lcode):
        self.tbid = tbid
        self.lcode = lcode
        self.init_basic_parser(lcode)

    def __repr__(self):
        return '<Default config for %r with parsers %r>' %(
            self.tbid, self.basic_parsers
        )

    def init_basic_parser(self, lcode, n_parsers = 5):
        self.basic_parsers = []
        for embedding in ('contextualised', 'external', 'internal'):
            parsers = self.get_basic_parsers(lcode, embedding)
            if parsers:
                break
        if not parsers:
            print('Warning: No parser found that supports', lcode)
        for index in range(n_parsers):
            round_robin_choice = parsers[index % len(parsers)]
            self.basic_parsers.append(round_robin_choice)
        
    def get_basic_parsers(self, lcode, embedding):
        if embedding == 'contextualised':
            parsers = self.get_basic_parsers_with_contextualised_embedding()
        elif embedding == 'external':
            parsers = self.get_basic_parsers_with_external_embedding()
        elif embedding == 'internal':
            parsers = self.get_basic_parsers_with_internal_embedding()
        else:
            raise ValueError('Unsupported embedding type %s' %embedding)
        retval = []
        # filter for lcode
        for parser_name in parsers:
            parser_module = importlib.import_module(parser_name)
            if parser_module.supports(lcode):
                retval.append(parser_name)
        return retval

    def get_basic_parsers_with_contextualised_embedding(self):
        return ['elmo_udpf']

    def get_basic_parsers_with_external_embedding(self):
        return ['fasttext_udpf']

    def get_basic_parsers_with_internal_embedding(self):
        return ['plain_udpf']

def main():
    options = Options()
    options.get_tasks_and_configs()

if __name__ == "__main__":
    main()
