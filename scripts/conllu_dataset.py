#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import collections
import hashlib
import os
import subprocess
import sys

import basic_dataset
import utilities

id_column = 0
token_column = 1
lemma_column = 2
pos_column = 3
morph_column = 5
head_column = 6
label_column = 7
enh_column = 8

def hex2base62(h):
    s = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    i = int(h, 16)
    if not i:
        return '0'
    digits = []
    while i:
        d = i % 62
        digits.append(s[d])
        i = int(i/62)
    return ''.join(digits)

class ConlluSentence(basic_dataset.Sentence):

    def __init__(self):
        basic_dataset.Sentence.__init__(self)
        self.rows = []
        self.token2row = []
        self.enh_token2row = []
        self.id2row = {}

    def __getitem__(self, index):
        return self.rows[self.token2row[index]]

    def collect_labels(self, labelset, column):
        for row in self:
            label = row[column]
            if label != '@@MISSING@@':
                labelset[label] = None

    def __len__(self):
        return len(self.token2row)

    def __repr__(self):
        return '<ConlluSentence %r %r>' %(
            self.token2row,
            self.rows,
        )

    def truncate_tokens(self, truncate_long_tokens = None, **kwargs):
        if not truncate_long_tokens:
            return
        for row_index in self.token2row:
            row = self.rows[row_index]
            token = row[1]
            if len(token) > truncate_long_tokens:
                fingerprint = hex2base62(hashlib.sha256(token).hexdigest())[:12]
                row[1] = 'tt:' + fingerprint
                row[2] = 'tt:' + fingerprint

    def append(self, line):
        fields = line.split('\t')
        if not fields:
            raise ValueError('cannot append empty line to conllu sentence')
        # remove linebreak from last cell
        fields[-1] = fields[-1].rstrip()
        #
        r_index = len(self.rows)
        self.rows.append(fields)
        # check for sentence ID
        if line.startswith('#') \
        and line.replace(' ', '').startswith('#sent_id='):
            eqpos = line.find('=')
            self.sent_id = line[eqpos+1:].strip()
        # check whether this is a UD token
        token_id = fields[id_column]
        if not token_id.startswith('#'):
            self.id2row[token_id] = r_index
        if token_id.startswith('#') \
        or '-' in token_id:
            return
        # record enhanced UD token
        self.enh_token2row.append(r_index)
        if '.' in token_id:
            return
        # record UD token
        self.token2row.append(r_index)

    def write(self, f_out, remove_comments = False):
        for row in self.rows:
            if remove_comments and row[0].startswith('#'):
                continue
            # incomplete sentences should be completed by the caller,
            # e.g. save_to_file(), see basic_dataset.SentenceCompleter
            if '@@MISSING@@' in row:
                for column, cell in enumerate(row):
                    if column:
                        f_out.write('\t')
                    if column < 2:
                        f_out.write(cell)
                    elif cell == '@@MISSING@@':
                        f_out.write('_')
                    else:
                        f_out.write(cell)
            else:
                f_out.write('\t'.join(row))
            f_out.write('\n')
        f_out.write('\n')

    def clone(self):
        copy = ConlluSentence()
        for row in self.rows:
            copy.append('\t'.join(row))
        return copy

    def is_missing(self, index, column):
        return self[index][column] == '@@MISSING@@'

    def set_label(self, index, column, new_label):
        self[index][column] = new_label

    def possible_labels(self, index, column):
        if column != head_column:
            # See SentenceCompleter for the normal way
            # to supply possible labels.
            raise ValueError('Sentence does not know labels for column %d.' %column)
        retval = list(range(len(self)+1))
        del retval[index+1]
        return map(lambda x: '%d' %x, retval)

    def unset_label(self, index, column):
        self[index][column] = '@@MISSING@@'

    def get_vector_representation(self):
        # TODO: return 4096 dimensional vector
        # counting each n-gram hash value
        raise NotImplementedError

class ConlluDataset(basic_dataset.Dataset):

    def __init__(self):
        basic_dataset.Dataset.__init__(self)

    def __repr__(self):
        return '<Conllu with %d sentences>' %(
            len(self),
        )

    def usable(self, sentence, max_token_bytes = None, **kwargs):
        if not max_token_bytes:
            return True
        for token_row in sentence:
            if len(token_row[1]) > max_token_bytes:
                return False
        return True

    def read_sentence(self, f_in):
        sentence = None
        while True:
            line = f_in.readline()
            if not line:
                if sentence is not None:
                    raise ValueError('unexpected end of file in conllu sentence')
                break
            elif sentence is None:
                sentence = ConlluSentence()
            if line.isspace():
                break
            sentence.append(line)
        if sentence is not None and 'truncate_long_tokens' in self.load_kwargs:
            sentence.truncate_tokens(**self.load_kwargs)
        return sentence

    def write_sentence(self, f_out, sentence, remove_comments = False):
        sentence.write(f_out, remove_comments)


def evaluate(prediction_path, gold_path, options, outname = None, reuse_nonemtpty = True):
    if not outname:
        outname = prediction_path[:-7] + '.eval.txt'
    if reuse_nonemtpty \
    and os.path.exists(outname) \
    and os.path.getsize(outname) > 0:
        return get_score_from_eval_txt(outname)
    # collapse enhanced dependencies as required by shared task eval script
    collapsed_dir = '%s/collapsed' %options.predictdir
    utilities.makedirs(collapsed_dir)
    collapsed = {}
    for what, input_conllu in [
        ('system', prediction_path),
        ('gold',   gold_path),
    ]:
        _, filename = input_conllu.rsplit('/', 1)
        collapsed_conllu = '%s/%s' %(collapsed_dir, filename)
        collapsed[what] = collapsed_conllu
        if not os.path.exists(collapsed_conllu):
            command = []
            command.append('scripts/wrapper-collapse-empty-nodes.sh')
            command.append(input_conllu)
            command.append(collapsed_conllu)
            if options.debug:
                print('Collapsing %s file...' %what)
            sys.stderr.flush()
            sys.stdout.flush()
            subprocess.call(command)
        elif options.debug:
            print('Re-using collapsed %s file' %what)
    # run official shared task script evaluation script
    # on collapsed files
    command = []
    command.append('scripts/iwpt20_xud_eval.py') # %os.environ['PRJ_DIR'])
    command.append('--output')
    command.append(outname)
    command.append('--verbose')
    command.append(collapsed['gold'])
    command.append(collapsed['system'])
    if options.debug:
        print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)
    return get_score_from_eval_txt(outname)

def get_score_from_eval_txt(outname, metric = 'ELAS'):
    score = (0.0, 'N/A')
    with open(outname, 'rb') as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            # [0]       [1] [2]         [3] [4]         [5] [6]         [7] [8]
            # LAS        |  79.756753596 |  79.756753596 |  79.756753596 |  79.756753596
            if fields[0] == metric:
                score = fields[6]
                score = (float(score), score)
                break
    return score

def get_tbname(tbid, treebank_dir, tbmapfile = None):
    if not tbmapfile:
        candidate_file = '%s/tbnames.tsv' %treebank_dir
        if os.path.exists(candidate_file):
            tbmapfile = candidate_file
        elif 'UD_TBNAMES' in os.environ:
            tbmapfile = os.environ['UD_TBNAMES']
    if tbmapfile:
        f = open(tbmapfile, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            if tbid == fields[1]:
                f.close()
                return fields[0]
        f.close()
        raise ValueError('TBID %r not found in %r' %(tbid, tbmapfile))
    if treebank_dir:
        # scan treebank folder
        tcode = tbid.split('_')[1]
        for entry in os.listdir(treebank_dir):
            # is this a good candidate entry?
            if entry.startswith('UD_') and tcode in entry.lower():
                # look inside
                if os.path.exists('%s/%s/%s-ud-test.conllu' %(
                    treebank_dir, entry, tbid
                )):
                    return entry
        raise ValueError('TBID %r not found in %r (must have test set)' %(tbid, treebank_dir))
    raise ValueError('TBID %r not found (need map file or treebank dir)' %tbid)

def load_conll2017(
    lcode, name, dataset_basedir, mode = 'map',
    **kwargs
):
    if not dataset_basedir:
        dataset_basedir = os.environ['CONLL2017_DIR']
    # scan the dataset folder
    datasets = []
    for entry in os.listdir(dataset_basedir):
        index = 0
        while True:
            filename = '%s/%s/%s-%s-%03d.conllu' %(
                dataset_basedir, entry, lcode, name, index
            )
            if os.path.exists(filename):
                data = basic_dataset.load_or_map_from_filename(
                    ConlluDataset(), filename, mode,
                    **kwargs
                )
                datasets.append(data)
                index += 1
            else:
                break
        if index:
            print('Loaded %d CoNLL-2017 file(s) for %s %s' %(index, lcode, name))
            # no need to check for another folder with files for
            # the same lcode
            break
    return basic_dataset.Concat(datasets)

def load(dataset_id,
    tbname = None,
    tbmapfile = None,
    load_tr = True, load_dev = True, load_test = True,
    mode = 'map',
    dataset_basedir = None,
    only_get_path = False,
    **kwargs
):
    # correct type of args passed with --load-data-keyword
    for key in ('max_token_bytes', 'truncate_long_tokens'):
        if key in kwargs:
            value = kwargs[key]
            if value and type(value) is str:
                kwargs[key] = int(value)
    # CoNNL 2017 data sets
    lcode = dataset_id.split('_')[0]
    if dataset_id.endswith('_cc17'):
        data = load_conll2017(
            lcode, 'common_crawl', dataset_basedir, mode,
            **kwargs
        )
        return data, None, None
    if dataset_id.endswith('_wp17'):
        data = load_conll2017(
            lcode, 'wikipedia', dataset_basedir, mode,
            **kwargs
        )
        return data, None, None
    # UD data sets
    tbid = dataset_id
    tr, dev, test = None, None, None
    if dataset_basedir:
        treebank_dir = dataset_basedir
    else:
        treebank_dir = os.environ['UD_TREEBANK_DIR']
    if not tbname:
        tbname = get_tbname(tbid, treebank_dir, tbmapfile)
    if load_tr:
        filename = '%s/%s/%s-ud-train.conllu' %(treebank_dir, tbname, tbid)
        if only_get_path:
            tr = filename
        elif os.path.exists(filename):
            tr = basic_dataset.load_or_map_from_filename(
                ConlluDataset(), filename, mode,
                **kwargs
            )
        else:
            print('Warning: %r not found' %filename)
    if load_dev:
        filename = '%s/%s/%s-ud-dev.conllu' %(treebank_dir, tbname, tbid)
        if only_get_path:
            dev = filename
        elif os.path.exists(filename):
            dev = basic_dataset.load_or_map_from_filename(
                ConlluDataset(), filename, mode,
                **kwargs
            )
        else:
            print('Warning: %r not found' %filename)
    if load_test:
        filename = '%s/%s/%s-ud-test.conllu' %(treebank_dir, tbname, tbid)
        if only_get_path:
            test = filename
        elif os.path.exists(filename):
            test = basic_dataset.load_or_map_from_filename(
                ConlluDataset(), filename, mode,
                **kwargs
            )
        else:
            print('Warning: %r not found' %filename)
    return tr, dev, test

def new_empty_set():
    return ConlluDataset()

def get_target_columns():
    return [2,3,4,5,6,7,8]

def get_filename_extension():
    ''' recommended extension for output files '''
    return '.conllu'

def combine(
    prediction_paths, output_path,
    options,
    prune_labels = False,
    combiner_dir = None, seed = '42'
):
    ''' combine (ensemble) the given predictions
        into a single prediction
    '''
    if not combiner_dir:
        combiner_dir = os.environ['CONLLU_COMBINER_DIR']
    command = []
    command.append('%s/parser.py' %combiner_dir)
    command.append('--outfile')
    command.append(output_path)
    command.append('--overwrite')
    if prune_labels:
        command.append('--prune-labels')
    command.append('--seed')
    command.append(seed)
    for prediction_path in prediction_paths:
        command.append(prediction_path)
    if options.debug:
        print('Running', command)
        sys.stderr.flush()
        sys.stdout.flush()
    subprocess.call(command)

class SentenceFilter:

    def __init__(self, max_token_bytes = None):
        self.max_token_bytes = max_token_bytes

    def __call__(self, sentence):
        ''' returns True if the sentence should be skipped '''
        if not self.max_token_bytes:
            return False
        for token_row in sentence:
            if len(token_row[1]) > self.max_token_bytes:
                return True
        return False

def get_filter(**kwargs):
    for key in ('max_token_bytes',):
        try:
            value = kwargs[key]
            if value and type(value) == str:
                value = int(value)
            else:
                value = 0
            kwargs[key] = value
        except KeyError:
            pass
    return SentenceFilter(**kwargs)

def main():
    import random
    import sys
    if len(sys.argv) < 2 or sys.argv[1][:3] in ('-h', '--h'):
        print('usage: $0 $NUMBER_SENTENCES {load|map} {dropoutsample|shuffle} < in.conllu > out.conllu')
        sys.exit(1)
    max_sentences = int(sys.argv[1])
    mode = sys.argv[2]
    dataset = ConlluDataset()
    dataset.load_or_map_file(sys.stdin, max_sentences, mode)
    if sys.argv[3] == 'shuffle':
        dataset.shuffle(random)
        dataset.save_to_file(sys.stdout)
        return
    dropout = basic_dataset.SentenceDropout(random,
            [pos_column, head_column, label_column],
            [0.2,        0.8,         0.5]
    )
    sample = basic_dataset.Sample(dataset, random, sentence_modifier = dropout)
    sample.save_to_file(sys.stdout)

if __name__ == "__main__":
    main()

