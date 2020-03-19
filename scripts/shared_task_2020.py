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

import utilities

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

    --skip-training  NAME   Skip training models for component NAME.
                            Can be one of segmenter, basic_parser and
                            enhanced_parser, and can be repeated to skip
                            training for more than one component.
                            Specifying a space- or colon-separated list
                            of components is also supported.
                            Skipping can be restricted to a subcompontent
                            with a dot, e.g. "basic_parser.plain_udpf".
                            (Default: train all missing models)

    --workdir  DIR          Path to working directory
                            (Default: data)

    --taskdir  DIR          Path to shared task data containing a UD-like
                            folder structure with training and dev files.
                            With --final-test, we assume that test files
                            follow the naming and structure of the dev data
                            and can be merged into the train-dev folder
                            provided by the shared task.
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
                            not exceed 327, assuming no more than 67 models
                            per treebank are trained, or 326 otherwise.
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
        self.modules_not_to_train = set()
        self.workdir    = 'data'
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
            elif option == '--skip-training':
                for name in sys.argv[1].replace(':', ' ').split():
                    self.modules_not_to_train.add(name)
                del sys.argv[1]
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
        self.prediction_tasks = []
        self.training_data = []
        self.configs = {}
        prepare_configs = []
        for tbname in os.listdir(self.taskdir):
            if not tbname.startswith('UD_'):
                if self.verbose:
                    print('Skipping non-UD entry in taskdir:', tbname)
                continue
            # Scan the treebank folder
            tbdir = '/'.join((self.taskdir, tbname))
            tbid_needs_models = False
            tbid, lcode, data_type = None, None, None
            for filename in os.listdir(tbdir):
                path = '/'.join((tbdir, filename))
                if '-ud-' in filename:
                    tbid, _, suffix = filename.rsplit('-', 2)
                    data_type = suffix.split('.')[0]
                if filename == 'tbid.txt':
                    f = open('%s/%s' %(tbdir, filename), 'r')
                    tbid = f.readline().strip()
                    f.close()
                    tbid_needs_models = True
                if tbid:
                    lcode = tbid.split('_')[0]
                    if '-' in tbid:
                        tbid = tbid.replace('-', '_')
                        if self.verbose:
                            print('Dash in TBID replaced with underscore')
                if filename.endswith('-ud-dev.txt') \
                or filename.endswith('-ud-test.txt'):
                    # found a prediction task
                    tbid_needs_models = True
                    self.prediction_tasks.append((path, tbid, lcode, data_type))
                if filename.endswith('-ud-train.conllu'):
                    # found training data
                    self.training_data.append((path, tbid, lcode))
            if tbid is None:
                print('Warning: No TBID found for %s. Please add `tbid.txt`.' %tbname)
            if not tbid_needs_models:
                continue
            # postpone config creation to when all folders have been read
            prepare_configs.append((tbid, lcode))
        prepare_configs.sort()
        for tbid, lcode in prepare_configs:
            if tbid in self.configs:
                print('Warning: Skipping duplicate TBID', tbid)
                continue
            if self.debug:
                print('Preparing configurations for TBID', tbid)
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
            # create all config variants for this tbid
            # (we will produce exactly one prediction for each
            # test set and config variant)
            config_variants = config_class(
                tbid, lcode,
                task_training_data = self.training_data,
                debug = self.debug,
            ).get_variants()
            configs_for_tbid = []
            for variant in config_variants:
                if self.debug:
                    print('Preparing variant %s for TBID %s' %(variant, tbid))
                configs_for_tbid.append(config_class(
                    tbid, lcode,
                    variant = variant,
                    task_training_data = self.training_data,
                    init_seed = self.init_seed,
                    debug = self.debug,
                ))
            self.configs[tbid] = configs_for_tbid
        if self.debug:
            print('prediction_tasks:', len(self.prediction_tasks))
            for item in sorted(self.prediction_tasks):
                print('\t', item)
            print('training_data:', len(self.training_data))
            for item in sorted(self.training_data):
                print('\t', item)
            print('configs:', len(self.configs))
            for key in sorted(list(self.configs.keys())):
                items = self.configs[key]
                print('\t', key, len(items))
                for item in sorted(items):
                    print('\t\t', item)

    def train_missing_models(self):
        tasks = []
        for tbid in sorted(list(self.configs.keys())):
            for config in self.configs[tbid]:
                if not config.is_operational():
                    print('Not training %r as it is not operational' %config)
                    continue
                if not config.skip(self.modules_not_to_train):
                    tasks += config.train_missing_models()
                else:
                    print('Not training %r as user requested to skip it' %config)
        utilities.wait_for_tasks(tasks)

class Config_default:

    def __init__(self, tbid, lcode,
        variant = None, task_training_data = None,
        debug = False,
        init_seed = '101',
    ):
        self.tbid = tbid
        self.lcode = lcode
        self.variant = variant
        self.task_training_data = task_training_data
        self.init_seed = init_seed
        self.debug = debug
        self.segmenter = None
        self.basic_parsers = []
        self.enhanced_parser = None
        if variant is not None:
            self.init_segmenter()
            self.init_basic_parsers()
            self.init_enhanced_parser()

    def __repr__(self):
        return '<Default config for %r with variant %r>' %(
            self.tbid, self.variant
        )

    def is_operational(self):
        if self.segmenter is None:
            return False
        if not self.basic_parsers:
            return False
        if self.enhanced_parser is None:
            return False
        return True

    def get_variants(self):
        # Sub-classes can add parameters to the module name separated by a
        # colon (':'), e.g. to get variants that use the same module for
        #       a component but different training data, e.g.
        #       udpipe_standard with english-ewt and
        #       udpipe_standard with english-gum
        basic_parser_ensemble_size = self.get_basic_parser_ensemble_size()
        for segmenter in self.get_segmenters():
            basic_parsers = list(self.get_basic_parsers())
            # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
            n_basic_parsers = len(basic_parsers)
            restrict_to_one = False
            if n_basic_parsers > 1:
                # TODO: prune list of parser to k-best parsers according to dev results
                found_dev_results = False
                if found_dev_results:
                    raise NotImplementedError
                else:
                    print('Too many parsers. Two pass mode needed.')
                    print('Individual parsers will be evaluated now.')
                    print('Then run again to test combinations of the best parsers.')
                    restrict_to_one = True
                    combination_indices = []
                    for i in range(n_basic_parsers):
                        combination_indices.append(1<<i)
            else:
                combination_indices = range(1, 1 << n_basic_parsers)
            for combination_index in combination_indices:
                combination = [basic_parsers[j]
                    for j in range(n_basic_parsers)
                    if (combination_index & (1 << j))
                ]
                if len(combination) > basic_parser_ensemble_size:
                    continue
                if restrict_to_one and len(combination) > 1:
                    continue
                for enhanced_parser in self.get_enhanced_parsers():
                    yield (segmenter, combination, enhanced_parser)

    def get_segmenter_names(self):
        return ['udpipe_standard', 'udpipe_augmented', 'udpipe_polyglot', 'uusegmenter']

    def get_basic_parser_names(self):
        return ['elmo_udpf', 'fasttext_udpf', 'plain_udpf', 'udify']

    def get_enhanced_parser_names(self):
        return ['enhanced_parser']

    def get_basic_parser_ensemble_size(self):
        return 3

    def get_segmenters(self):
        if self.debug:
            print('Segmenters for', self.tbid)
        return self.filter_by_lcode_and_add_datasets(
            self.get_segmenter_names()
        )

    def get_basic_parsers(self):
        if self.debug:
            print('Basic parsers for', self.tbid)
        return self.filter_by_lcode_and_add_datasets(
            self.get_basic_parser_names()
        )

    def get_enhanced_parsers(self):
        if self.debug:
            print('Enhanced parsers for', self.tbid)
        return self.filter_by_lcode_and_add_datasets(
            self.get_enhanced_parser_names()
        )

    def filter_by_lcode_and_add_datasets(self, module_names):
        if self.debug:
            print('\tQuerying datasets for', self.tbid)
            sys.stdout.flush()
        datasets = list(self.get_dataset_names())
        datasets.sort()
        if self.debug:
            print('\t-->', datasets)
        retval = []
        for module_name in module_names:
            if self.debug:
                print('\t\tImporting', module_name)
                sys.stdout.flush()
            try:
                my_module = importlib.import_module(module_name)
            except ImportError:
                print('Warning: module %r not available, skipping' %module_name)
                continue
            if my_module.supports_lcode(self.lcode):
                if self.debug:
                    print('\t\tLanguage %s is supported' %self.lcode)
                if not datasets:
                    if my_module.has_default_model_for_lcode(self.lcode):
                        retval.append((0, module_name))
                else:
                    tbid_combinations_covered = {}
                    all_tbids = set()
                    # find number of tbids
                    for dataset in datasets:
                        _, tbid = dataset.split('.', 1)
                        all_tbids.add(tbid)
                    n_all_tbids = len(all_tbids)
                    # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
                    n_datasets = len(datasets)
                    for combination_index in range(1, 1 << n_datasets):
                        combination = [datasets[j]
                            for j in range(n_datasets)
                            if (combination_index & (1 << j))
                        ]
                        if self.debug:
                            print('\t\t\tChecking dataset combination', combination)

                        contains_ud25_data = False
                        contains_task_data = False
                        _, first_tbid = combination[0].split('.', 1)
                        first_lcode = first_tbid.split('_')[0]
                        contains_target_lcode = False
                        duplicate_tbid = False
                        data_lcodes = set()
                        data_tbids = set()
                        for dataset_name in combination:
                            if dataset_name.startswith('ud25.'):
                                contains_ud25_data = True
                            if dataset_name.startswith('task.'):
                                contains_task_data = True
                            data_source, tbid = dataset_name.split('.', 1)
                            lcode = tbid.split('_')[0]
                            if lcode == self.lcode:
                                contains_target_lcode = True
                            if tbid in data_tbids:
                                duplicate_tbid = True
                            data_lcodes.add(lcode)
                            data_tbids.add(tbid)
                        prio1 = len(data_tbids)
                        prio2 = n_all_tbids - prio1
                        priority = min(2*prio1-1, 2*prio2)
                        if duplicate_tbid:
                            if self.debug:
                                print('\t\t\t--> duplicate tbid, skipping')
                            continue
                        tbid_combi = tuple(sorted(data_tbids))
                        if not contains_target_lcode:
                            # dataset combination has no data with the
                            # target lcode --> skip
                            if self.debug:
                                print('\t\t\t--> no matching lcode, skipping')
                            continue
                        is_polyglot = len(data_lcodes) > 1
                        #if is_polyglot:
                        #    TODO: do we want to check that all languages are supported,
                        #          e.g. word embeddings are available, or do we assume
                        #          that modules that support polyglot training use internal
                        #          word embeddings only?
                        if len(combination) == 1 \
                        and contains_ud25_data   \
                        and my_module.has_ud25_model_for_tbid(combination[0][5:]):
                            if tbid_combi in tbid_combinations_covered:
                                if self.debug:
                                    print('\t\t\t-->not adding as already have', tbid_combinations_covered[tbid_combi])
                            else:
                                retval.append((priority, ':'.join((module_name, combination[0]))))
                                tbid_combinations_covered[tbid_combi] = combination
                                if self.debug:
                                    print('\t\t\t--> added data as module has a ud25 model ready')
                        if len(combination) == 1 \
                        and contains_task_data   \
                        and my_module.has_task_model_for_tbid(combination[0][5:]):
                            if tbid_combi in tbid_combinations_covered:
                                if self.debug:
                                    print('\t\t\t-->not adding as already have', tbid_combinations_covered[tbid_combi])
                            else:
                                retval.append((priority, ':'.join((module_name, combination[0]))))
                                tbid_combinations_covered[tbid_combi] = combination
                                if self.debug:
                                    print('\t\t\t--> added data as module has a task model ready')
                        elif my_module.can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
                            if tbid_combi in tbid_combinations_covered:
                                if self.debug:
                                    print('\t\t\t-->not adding as already have', tbid_combinations_covered[tbid_combi])
                            else:
                                retval.append((priority, ':'.join((module_name, '+'.join(combination)))))
                                tbid_combinations_covered[tbid_combi] = combination
                                if self.debug:
                                    print('\t\t\t--> added data as module can train on it')
                        elif self.debug:
                            print('\t\t\tNo suitable data found for')
                            print('\t\t\t * contains_ud25_data', contains_ud25_data)
                            print('\t\t\t * contains_task_data', contains_task_data)
                            print('\t\t\t * is_polyglot', is_polyglot)
            elif self.debug:
                print('\t\tLanguage %s is not supported' %self.lcode)
        retval.sort()
        if len(retval) > 2:
            print('\tPruning too long list of modules', retval)
            retval = retval[:2]
        # remove priority
        retval = map(lambda x: x[1], retval)
        if self.debug:
            print('\tFinished checking modules for', self.tbid)
            print('\t-->', retval)
            sys.stdout.flush()
        return retval

    def get_dataset_names(self):
        ''' subclasses should extend this list to include
            appropriate ud25 data
        '''
        for path, tbid, lcode in self.task_training_data:
            if lcode == self.lcode:
                yield 'task.' + tbid

    def init_segmenter(self):
        self.segmenter = self.variant[0]

    def init_basic_parsers(self):
        self.basic_parsers = []
        parsers = self.variant[1]
        if not parsers:
            print('Warning: No parser found that supports', self.lcode)
        for index in range(self.get_basic_parser_ensemble_size()):
            round_robin_choice = parsers[index % len(parsers)]
            self.basic_parsers.append(round_robin_choice)

    def init_enhanced_parser(self):
        self.enhanced_parser = self.variant[2]

    def skip(self, skip_list):
        segmenter, basic_parsers, enhanced_parser = self.variant
        if self.p_skip('segmenter', segmenter, skip_list):
            return True
        for basic_parser in basic_parsers:
            if self.p_skip('basic_parser', basic_parser, skip_list):
                return True
        if self.p_skip('enhanced_parser', enhanced_parser, skip_list):
            return True
        return False

    def p_skip(self, category, module_name, skip_list):
        if category in skip_list:
            return True
        combined_name = '.'.join((category, module_name))
        if combined_name in skip_list:
            return True
        return False

    def train_missing_models(self):
        tasks = self.train_missing_segmenter_model()
        tasks += self.train_missing_basic_parser_models()
        tasks += self.train_missing_enhanced_parser_model()
        return tasks

    def train_missing_segmenter_model(self):
        segmenter_module, datasets = self.variant[0].split(':', 1)
        segmenter = importlib.import_module(segmenter_module)
        if self.debug:
            print('Training %s with seed %s and data %s' %(
                segmenter_module, self.init_seed, datasets,
            ))
        tasks = []
        tasks.append(segmenter.train_model_if_missing(
            self.lcode,
            self.init_seed,
            datasets,

        ))
        return tasks

    def train_missing_basic_parser_models(self):
        tasks = []
        for p_index, p_name in enumerate(self.variant[1]):
            p_module_name, datasets = p_name.split(':', 1)
            basic_parser = importlib.import_module(p_module_name)
            init_seed = '%s%02d' %(self.init_seed, p_index)
            if self.debug:
                print('Training %s with seed %s and data %s' %(
                    p_module_name, init_seed, datasets,
                ))
            tasks.append(basic_parser.train_model_if_missing(
                self.lcode,
                init_seed,
                datasets,
            ))

    def train_missing_enhanced_parser_model(self):
        enhanced_module, datasets = self.variant[2].split(':', 1)
        enhanced_parser = importlib.import_module(enhanced_module)
        if self.debug:
            print('Training %s with seed %s and data %s' %(
                enhanced_module, self.init_seed, datasets,
            ))
        enhanced_parser.train_model_if_missing(
            self.lcode,
            self.init_seed,
            datasets,
        )

class Config_cs(Config_default):

    def get_dataset_names(self):
        tbids_covered = set()
        for dataset in Config_default.get_dataset_names(self):
            if dataset.startswith('task.'):
                tbid = dataset.split('.')[1]
                tbids_covered.add(tbid)
                yield dataset
        for tbid in ('cs_cac', 'cs_pdt'):
            #if tbid not in tbids_covered:
            yield 'ud25.' + tbid

class Config_en(Config_default):

    def get_dataset_names(self):
        tbids_covered = set()
        for dataset in Config_default.get_dataset_names(self):
            if dataset.startswith('task.'):
                tbid = dataset.split('.')[1]
                tbids_covered.add(tbid)
                yield dataset
        for tbid in ('en_ewt', 'en_gum', 'en_lines', 'en_partut'):
            #if tbid not in tbids_covered:
            yield 'ud25.' + tbid

def main():
    options = Options()
    options.get_tasks_and_configs()
    options.train_missing_models()

if __name__ == "__main__":
    main()
