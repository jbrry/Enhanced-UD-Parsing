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
from collections import defaultdict

import conllu_dataset
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
                            The special name ALL is expanded to all 3
                            component types.
                            Skipping can be restricted to a subcompontent
                            with a dot, e.g. "basic_parser.plain_udpf".
                            (Default: train all missing models)

    --training-only         Quit after training tasks have been submitted.
                            (Default: Wait for tasks to finish and continue
                            with prediction and development set
                            evaluation.)

    --workdir  DIR          Path to working directory.
                            Note that some modules require absolute paths.
                            (Default: $PWD/data)

    --taskdir  DIR          Path to shared task data containing a UD-like
                            folder structure with training and dev files.
                            With --final-test, we assume that test files
                            follow the naming and structure of the dev data
                            and can be merged into the train-dev folder
                            provided by the shared task.
                            (Default: workdir + /train-dev)

    --ud25dir  DIR          Path to the UD v2.5 folder
                            (Default: workdir + /ud-treebanks-v2.5)

    --tempdir  DIR          Path to directory for intermediate files
                            (Default: workdir + /temp)

    --modeldir  DIR         Path to model directory
                            (Default: workdir + /models)

    --predictdir  DIR       Path to prediction directory
                            (Default: workdir + /predictions)

    --init-seed  STRING     Initialise random number generator with STRING.
                            If STRING is empty, use system seed.
                            Will be passed to basic parsers for training
                            with a
                            2-digit suffix. If, for example, the parser can
                            only accepts seeds up to 32767 this seed must
                            not exceed 327, assuming no more than 67 models
                            per treebank are trained, or 326 otherwise.
                            (Default: 42)

    --verbose               More detailed log output

""")

    def read_options(self):
        self.final_test = False
        self.modules_not_to_train = set()
        self.training_only = False
        self.workdir    = None
        self.taskdir    = None
        self.ud25dir    = None
        self.tempdir    = None
        self.modeldir   = None
        self.predictdir = None
        self.init_seed  = 42
        self.verbose    = False
        self.debug      = True
        while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
            option = sys.argv[1]
            option = option.replace('_', '-')
            del sys.argv[1]
            if option in ('--help', '-h'):
                return True
            elif option == '--final-test':
                self.final_test = True
            elif option == '--skip-training':
                names = sys.argv[1].replace(':', ' ').split()
                if 'ALL' in names:
                    names = 'segmenter basic_parser enhanced_parser'.split()
                for name in names:
                    self.modules_not_to_train.add(name)
                del sys.argv[1]
            elif option == '--workdir':
                self.workdir = sys.argv[1]
                del sys.argv[1]
            elif option == '--taskdir':
                self.taskdir = sys.argv[1]
                del sys.argv[1]
            elif option == '--tempdir':
                self.tempdir = sys.argv[1]
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
            elif option == '--training-only':
                self.training_only = True
            elif option == '--verbose':
                self.verbose = True
            elif option == '--debug':
                self.debug = True
            else:
                print('Unsupported or not yet implemented option %s' %option)
                return True
        if self.debug:
            self.verbose = True
        if self.workdir is None:
            self.workdir = '/'.join((os.getcwd(), 'data'))
        if self.taskdir is None:
            self.taskdir = '/'.join((self.workdir, 'train-dev'))
        if self.ud25dir is None:
            self.ud25dir = '/'.join((self.workdir, 'ud-treebanks-v2.5'))
        if self.tempdir is None:
            self.tempdir = '/'.join((self.workdir, 'temp'))
        if self.modeldir is None:
            self.modeldir = '/'.join((self.workdir, 'models'))
        if self.predictdir is None:
            self.predictdir = '/'.join((self.workdir, 'predictions'))
        if len(sys.argv) != 1:
            return True
        return False

    def scan_ud_or_task_folder(self, folder_path, folder_type):
        retval = {}
        for tbname in os.listdir(folder_path):
            if not tbname.startswith('UD_'):
                if self.verbose:
                    print('Skipping non-UD entry in %s: %s' %(folder_type, tbname))
                continue
            # Scan the treebank folder
            tbdir = '/'.join((folder_path, tbname))
            tbid_needs_models = False
            tbid = None
            training_data = None
            prediction_tasks = []
            dev_conllu = None
            for filename in os.listdir(tbdir):
                path = '/'.join((tbdir, filename))
                new_tbid = None
                dataset_type, dataset_format = None, None
                if '-ud-' in filename:
                    new_tbid, _, suffix = filename.rsplit('-', 2)
                    dataset_type, dataset_format = suffix.split('.')
                elif filename == 'tbid.txt':
                    f = open('%s/%s' %(tbdir, filename), 'r')
                    new_tbid = f.readline().strip()
                    f.close()
                    tbid_needs_models = True
                if new_tbid:
                    if '-' in new_tbid:
                        if self.verbose:
                            print('Dash in TBID %s replaced with underscore' %new_tbid)
                        new_tbid = new_tbid.replace('-', '_')
                    if tbid is not None and new_tbid != tbid:
                        raise ValueError('Mismatching TBIDs in %s' %tbdir)
                    tbid = new_tbid
                if dataset_format == 'txt' and dataset_type in ('dev', 'test'):
                    # found a prediction task
                    tbid_needs_models = True
                    prediction_tasks.append((path, dataset_type))
                if (dataset_type, dataset_format) == ('train', 'conllu'):
                    # found training data
                    training_data = path
                if (dataset_type, dataset_format) == ('dev', 'conllu'):
                    dev_conllu = path
            if tbid is None:
                print('Warning: No TBID found for %s %s. Please add `tbid.txt`.' %(folder_type, tbname))
            else:
                if tbid in retval:
                    print('Warning: Duplicate TBID %s causing information loss in %s' %(tbid, folder_path))
                retval[tbid] = (
                    tbdir, training_data, tbid_needs_models, prediction_tasks,
                    dev_conllu,
                )
        return retval

    def get_tasks_and_configs(self):
        self.task_treebanks = self.scan_ud_or_task_folder(
            self.taskdir, 'taskdir'
        )
        self.configs = {}
        for tbid in sorted(list(self.task_treebanks.keys())):
            tb_info = self.task_treebanks[tbid]
            _, _, tbid_needs_models, _, _ = tb_info
            if not tbid_needs_models:
                continue
            if self.debug:
                print('Preparing configurations for TBID', tbid)
                prediction_tasks = tb_info[3]
                print('%d prediction tasks:' %len(prediction_tasks))
                for item in sorted(prediction_tasks):
                    print('\t\t', item)
                training_data = tb_info[1]
                print('Training data:', training_data)
                dev_conllu = tb_info[4]
                print('Dev data:', dev_conllu)
            lcode = tbid.split('_')[0]
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
                options = self,
            ).get_variants()
            configs_for_tbid = []
            for variant in config_variants:
                if self.debug:
                    print('Preparing variant %s for TBID %s' %(variant, tbid))
                configs_for_tbid.append(config_class(
                    tbid, lcode,
                    options = self,
                    variant = variant,
                ))
            self.configs[tbid] = configs_for_tbid
        if self.debug:
            print('%d TBIDs configured' %len(self.configs))
            for key in sorted(list(self.configs.keys())):
                items = self.configs[key]
                print('\t', key, len(items))
                for item in sorted(items):
                    print('\t\t', item)

    def train_missing_models(self):
        if self.verbose:
            print('\n== Training Models ==\n')
        tasks = []
        self.in_progress = set()
        for tbid in sorted(list(self.configs.keys())):
            for config in self.configs[tbid]:
                if not config.is_operational():
                    print('Not training %r as it is not operational' %config)
                    continue
                if not config.skip(self.modules_not_to_train):
                    tasks += config.train_missing_models()
                else:
                    print('Not training %r as user requested to skip it' %config)
        if self.verbose:
            print('Submitted %d training task(s)' %(len(tasks)-tasks.count(None)))
        if self.training_only:
            return
        utilities.wait_for_tasks(tasks)

    def scan_ud25(self):
        self.ud25_treebanks = self.scan_ud_or_task_folder(
            self.ud25dir, 'UD v2.5'
        )

    def predict_and_evaluate(self):
        if self.verbose:
            print('\n== Prediction and Evaluation ==\n')
        segment_dir = '/'.join((self.tempdir, 'segmented'))
        utilities.makedirs(segment_dir)
        basic_p_dir = '/'.join((self.tempdir, 'basic-parses'))
        utilities.makedirs(basic_p_dir)
        enhanced_dir = self.predictdir
        utilities.makedirs(enhanced_dir)
        for tbid in sorted(list(self.task_treebanks.keys())):
            prediction_tasks = self.task_treebanks[tbid][3]
            for input_path, prediction_dataset_type in prediction_tasks:
                print('\n=== %s ===\n' %input_path)
                if prediction_dataset_type == 'test' \
                and not self.final_test:
                    # no need to make prediction for test sets unless
                    # specifically requested with --final-test
                    continue
                tb_dir, text_filename = input_path.rsplit('/', 1)
                assert text_filename.startswith(tbid)
                assert text_filename.endswith('.txt')
                prediction_name = text_filename[:-4]
                for config in self.configs[tbid]:
                    print('\n==== %r ====\n' %config)
                    # segmentation
                    segmented_path = '%s/%s-for-%s.conllu' %(
                        segment_dir,
                        config.segmenter_id,
                        prediction_name
                    )
                    if not os.path.exists(segmented_path):
                        config.segment(input_path, segmented_path)
                    else:
                        print('Reusing existing segmentation')
                    if not os.path.exists(segmented_path):
                        print('Warning: Failure to produce %s' %segmented_path)
                        continue
                    # basic parsing
                    basic_p_path = '%s/%s-%s-for-%s.conllu' %(
                        basic_p_dir,
                        config.segmenter_id,
                        config.basic_parser_id,
                        prediction_name
                    )
                    if not os.path.exists(basic_p_path):
                        config.parse(segmented_path, basic_p_path)
                    else:
                        print('Reusing existing basic parse')
                    if not os.path.exists(basic_p_path):
                        print('Warning: Failure to produce %s' %basic_p_path)
                        continue
                    # enhanced parsing
                    enhanced_path = '%s/%s-%s-%s-for-%s.conllu' %(
                        enhanced_dir,
                        config.segmenter_id,
                        config.basic_parser_id,
                        config.enhanced_parser_id,
                        prediction_name
                    )
                    if not os.path.exists(enhanced_path):
                        config.enhance(basic_p_path, enhanced_path)
                    else:
                        print('Reusing existing enhanced parse')
                    if not os.path.exists(enhanced_path):
                        print('Warning: Failure to produce %s' %enhanced_path)
                        continue
                    # evaluate: gold file is assumed to be in the same folder
                    # as the input text file with .conllu instead of .txt
                    gold_path = '%s/%s.conllu' %(tb_dir, prediction_name)
                    conllu_dataset.evaluate(enhanced_path, gold_path, self)

class Config_default:

    def __init__(self, tbid, lcode, options,
        variant = None,
    ):
        self.tbid = tbid
        self.lcode = lcode
        self.variant = variant
        self.options = options
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
        for segmenter in self.get_segmenters():
            basic_parsers = list(self.get_basic_parsers())
            # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
            n_basic_parsers = len(basic_parsers)
            restrict_to_one = False
            if n_basic_parsers > 4:
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
                if restrict_to_one and len(combination) > 1:
                    continue
                for enhanced_parser in self.get_enhanced_parsers():
                    for ensemble_size in (3,5,7):
                        if len(combination) <= ensemble_size:
                            yield (segmenter, combination, enhanced_parser, ensemble_size)

    def get_segmenter_names(self):
        return ['udpipe_standard', 'udpipe_augmented', 'udpipe_polyglot', 'uusegmenter']

    def get_basic_parser_names(self):
        return ['elmo_udpf', 'udify', 'fasttext_udpf', 'plain_udpf',]

    def get_enhanced_parser_names(self):
        return ['copy_parse',]

    def get_basic_parser_ensemble_size(self):
        return self.variant[3]

    def get_segmenters(self):
        if self.options.debug:
            print('Segmenters for', self.tbid)
        return self.filter_by_lcode_and_add_datasets(
            self.get_segmenter_names()
        )

    def get_basic_parsers(self):
        if self.options.debug:
            print('Basic parsers for', self.tbid)
        return self.filter_by_lcode_and_add_datasets(
            self.get_basic_parser_names()
        )

    def get_enhanced_parsers(self):
        if self.options.debug:
            print('Enhanced parsers for', self.tbid)
        return self.filter_by_lcode_and_add_datasets(
            self.get_enhanced_parser_names()
        )

    def filter_by_lcode_and_add_datasets(self, module_names):
        if self.options.debug:
            print('\tQuerying datasets for', self.tbid)
            sys.stdout.flush()
        datasets = list(self.get_dataset_names())
        datasets.sort()
        if self.options.debug:
            print('\t-->', datasets)
        retval = []
        for module_name in module_names:
            if self.options.debug:
                print('\t\tImporting', module_name)
                sys.stdout.flush()
            try:
                my_module = importlib.import_module(module_name)
            except ImportError:
                print('Warning: module %r not available, skipping' %module_name)
                continue
            if my_module.supports_lcode(self.lcode):
                if self.options.debug:
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
                        if self.options.debug:
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
                            if self.options.debug:
                                print('\t\t\t--> duplicate tbid, skipping')
                            continue
                        tbid_combi = tuple(sorted(data_tbids))
                        if not contains_target_lcode:
                            # dataset combination has no data with the
                            # target lcode --> skip
                            if self.options.debug:
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
                                if self.options.debug:
                                    print('\t\t\t-->not adding as already have', tbid_combinations_covered[tbid_combi])
                            else:
                                retval.append((priority, ':'.join((module_name, combination[0]))))
                                tbid_combinations_covered[tbid_combi] = combination
                                if self.options.debug:
                                    print('\t\t\t--> added data as module has a ud25 model ready')
                        if len(combination) == 1 \
                        and contains_task_data   \
                        and my_module.has_task_model_for_tbid(combination[0][5:]):
                            if tbid_combi in tbid_combinations_covered:
                                if self.options.debug:
                                    print('\t\t\t-->not adding as already have', tbid_combinations_covered[tbid_combi])
                            else:
                                retval.append((priority, ':'.join((module_name, combination[0]))))
                                tbid_combinations_covered[tbid_combi] = combination
                                if self.options.debug:
                                    print('\t\t\t--> added data as module has a task model ready')
                        elif my_module.can_train_on(contains_ud25_data, contains_task_data, is_polyglot):
                            if tbid_combi in tbid_combinations_covered:
                                if self.options.debug:
                                    print('\t\t\t-->not adding as already have', tbid_combinations_covered[tbid_combi])
                            else:
                                retval.append((priority, ':'.join((module_name, '+'.join(combination)))))
                                tbid_combinations_covered[tbid_combi] = combination
                                if self.options.debug:
                                    print('\t\t\t--> added data as module can train on it')
                        elif self.options.debug:
                            print('\t\t\tNo suitable data found for')
                            print('\t\t\t * contains_ud25_data', contains_ud25_data)
                            print('\t\t\t * contains_task_data', contains_task_data)
                            print('\t\t\t * is_polyglot', is_polyglot)
            elif self.options.debug:
                print('\t\tLanguage %s is not supported' %self.lcode)
        retval.sort()
        if len(retval) > 5:
            print('\tPruning too long list of modules', retval)
            retval = retval[:5]
        # remove priority
        retval = map(lambda x: x[1], retval)
        if self.options.debug:
            print('\tFinished checking modules for', self.tbid)
            print('\t-->', retval)
            sys.stdout.flush()
        return retval

    def get_dataset_names(self):
        ''' subclasses should extend this list to include
            appropriate ud25 data
        '''
        for tbid in self.options.task_treebanks:
            lcode = tbid.split('_')[0]
            tb_info = self.options.task_treebanks[tbid]
            training_data = tb_info[1]
            if training_data and self.lcode == lcode :
                yield 'task.' + tbid

    def init_segmenter(self):
        self.segmenter = self.variant[0]
        segmenter_module, datasets = self.segmenter.split(':', 1)
        segmenter = importlib.import_module(segmenter_module)
        self.segmenter_id = segmenter.get_model_id(
            self.lcode,
            self.options.init_seed,
            datasets,
            self.options,
        )

    def init_basic_parsers(self):
        self.basic_parsers = []
        parsers = self.variant[1]
        if not parsers:
            print('Warning: No parser found that supports', self.lcode)
        ensemble_size = self.get_basic_parser_ensemble_size()
        p_name2p_index = defaultdict(lambda: 0)
        for index in range(ensemble_size):
            round_robin_choice = parsers[index % len(parsers)]
            p_index = p_name2p_index[round_robin_choice]
            p_name2p_index[round_robin_choice] = p_index + 1
            self.basic_parsers.append((round_robin_choice, p_index))
        # set short name of combination
        if len(parsers) > ensemble_size:
            parsers = parsers[:ensemble_size]
        name = []
        name.append('e%d' %ensemble_size)
        for parser in parsers:
            # TODO: cover case that the basic parser uses an off-the-shelf model
            #       that ignores the dataset and/or init seed
            name.append(parser.replace(':', '_').replace('.', '_'))
        self.basic_parser_id = '_'.join(name)

    def init_enhanced_parser(self):
        self.enhanced_parser = self.variant[2]
        enhanced_module, datasets = self.enhanced_parser.split(':', 1)
        enhanced_parser = importlib.import_module(enhanced_module)
        self.enhanced_parser_id = enhanced_parser.get_model_id(
            self.lcode,
            self.options.init_seed,
            datasets,
            self.options,
        )

    def skip(self, skip_list):
        segmenter, basic_parsers, enhanced_parser, _ = self.variant
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
        segmenter_module, datasets = self.segmenter.split(':', 1)
        segmenter = importlib.import_module(segmenter_module)
        if self.options.debug:
            print('Training %s with seed %s and data %s' %(
                segmenter_module, self.options.init_seed, datasets,
            ))
        tasks = []
        tasks.append(segmenter.train_model_if_missing(
            self.lcode,
            self.options.init_seed,
            datasets,
            self.options,
        ))
        return tasks

    def train_missing_basic_parser_models(self):
        tasks = []
        for p_name, p_index in self.basic_parsers:
            p_module_name, datasets = p_name.split(':', 1)
            basic_parser = importlib.import_module(p_module_name)
            init_seed = '%s%02d' %(self.options.init_seed, p_index)
            if self.options.debug:
                print('Training %s with seed %s and data %s' %(
                    p_module_name, init_seed, datasets,
                ))
            tasks.append(basic_parser.train_model_if_missing(
                self.lcode,
                init_seed,
                datasets,
                self.options,
            ))
        return tasks

    def train_missing_enhanced_parser_model(self):
        enhanced_module, datasets = self.enhanced_parser.split(':', 1)
        enhanced_parser = importlib.import_module(enhanced_module)
        if self.options.debug:
            print('Training %s with seed %s and data %s' %(
                enhanced_module, self.options.init_seed, datasets,
            ))
        tasks = []
        tasks.append(enhanced_parser.train_model_if_missing(
            self.lcode,
            self.options.init_seed,
            datasets,
            self.options,
        ))
        return tasks

    def segment(self, raw_text_input, conllu_output):
        segmenter_module, datasets = self.segmenter.split(':', 1)
        segmenter = importlib.import_module(segmenter_module)
        if self.options.debug:
            print('Predicting segmentation for %s using %s trained on %s' %(
                raw_text_input, segmenter_module, datasets,
            ))
        segmenter.predict(
            self.lcode, self.options.init_seed, datasets, self.options,
            raw_text_input, conllu_output,
        )

    def enhance(self, conllu_input, conllu_output):
        enhancer_module, datasets = self.enhanced_parser.split(':', 1)
        enhancer = importlib.import_module(enhancer_module)
        if self.options.debug:
            print('Enhancing parse for %s using %s trained on %s' %(
                conllu_input, enhancer_module, datasets,
            ))
        enhancer.predict(
            self.lcode, self.options.init_seed, datasets, self.options,
            conllu_input, conllu_output,
        )

    def parse(self, conllu_input, conllu_output):
        votesdir = '%s/basic-parses/votes' %self.options.tempdir
        utilities.makedirs(votesdir)
        ensemble_predictions = []
        _, vote_suffix = conllu_input.rsplit('/', 1)
        n_parses = len(self.basic_parsers)
        next_attempt = 1
        for parser_and_dataset, p_index in self.basic_parsers:
            parser_module, datasets = parser_and_dataset.split(':', 1)
            parser = importlib.import_module(parser_module)
            init_seed = '%s%02d' %(self.options.init_seed, p_index)
            conllu_individual_output = '%s/%s-%s-%s-%s' %(
                votesdir,
                parser_module,
                datasets.replace(':', '_').replace('-', '_'),
                init_seed,
                vote_suffix,
            )
            if os.path.exists(conllu_individual_output):
                action = 'Reusing'
                is_successful = True
            else:
                action = 'Predicting'
            if self.options.debug:
                print('%s basic parse %d of %d for %s using %s trained on %s with seed %s' %(
                    action,
                    next_attempt, n_parses,
                    conllu_input, parser_module, datasets,
                    init_seed,
                ))
            if action == 'Predicting':
                if '+' in datasets:
                    # multi-treebank mode
                    proxy_tbid = self.tbid
                else:
                    proxy_tbid = None
                is_successful = parser.predict(
                    self.lcode, init_seed, datasets, self.options,
                    conllu_input, conllu_individual_output,
                    proxy_tbid = proxy_tbid,
                )
            next_attempt += 1
            if is_successful:
                ensemble_predictions.append(conllu_individual_output)
        if self.options.debug:
            print('%d of %d basic parses are ready' %(len(ensemble_predictions), n_parses))
        if len(ensemble_predictions) < n_parses:
            print('Warning: missing basic parse(s) for ensemble --> skipping')
        elif len(ensemble_predictions) == 1:
            # just 1 parse --> no combination
            link_target = os.path.abspath(ensemble_predictions[0])
            os.symlink(link_target, conllu_output)
        else:
            # run combiner
            conllu_dataset.combine(ensemble_predictions, conllu_output, self.options)


class Config_with_more_datasets(Config_default):

    def get_dataset_names(self):
        tbids_covered = set()
        for dataset in Config_default.get_dataset_names(self):
            if dataset.startswith('task.'):
                tbid = dataset.split('.')[1]
                tbids_covered.add(tbid)
                yield dataset
        for tbid, allow_duplicate_tbid in self.get_additional_dataset_tbids():
            if allow_duplicate_tbid or tbid not in tbids_covered:
                yield 'ud25.' + tbid
                tbids_covered.add(tbid)

class Config_cs(Config_with_more_datasets):

    def get_additional_dataset_tbids(self):
        return [
            ('cs_cac', False),
            ('cs_pdt', False),
            ('cs_cltt', False),
        ]

class Config_en(Config_with_more_datasets):

    def get_additional_dataset_tbids(self):
        return [
            ('en_ewt', False),
            ('en_gum', False),
            ('en_lines', False),
            ('en_partut', False),
        ]

class Config_fr(Config_with_more_datasets):

    def get_additional_dataset_tbids(self):
        return [
            ('fr_gsd', False),
            ('fr_partut', False),
            ('fr_sequoia', False),
            ('fr_spoken', False),
        ]

class Config_it(Config_with_more_datasets):

    def get_additional_dataset_tbids(self):
        return [
            ('it_isdt', False),
            ('it_partut', False),
            ('it_postwita', False),
            ('it_twittiro', False),
            ('it_vit', False),
        ]

class Config_ru(Config_with_more_datasets):

    def get_additional_dataset_tbids(self):
        return [
            ('ru_syntagrus', False),
            ('ru_gsd', False),
            ('ru_taiga', False),
        ]

def main():
    options = Options()
    options.get_tasks_and_configs()
    options.scan_ud25()
    options.train_missing_models()  # respects --skip-training
    if options.training_only:
        return
    options.predict_and_evaluate()

if __name__ == "__main__":
    main()
