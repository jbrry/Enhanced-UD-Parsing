#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import hashlib
import os
import subprocess
import time

def get_score_stats(scores):
    scores.sort()
    min_score = scores[0]
    max_score = scores[-1]
    num_scores = len(scores)
    if (num_scores % 2):
        # odd number of scores
        median = scores[(num_scores-1)/2]
    else:
        # even number of scores
        median = (scores[num_scores/2-1] + scores[num_scores/2])/2.0
    backup = scores
    prune = int(0.025*len(scores))
    if prune:
        scores = scores[prune:-prune]
    score025 = scores[0]
    score975 = scores[-1]
    scores = backup
    prune = int(0.25*len(scores))
    if prune:
        scores = scores[prune:-prune]
    score250 = scores[0]
    score750 = scores[-1]
    return (min_score, score025, score250, median, score750, score975, max_score)

def float_with_suffix(size):
    multiplier = 1
    for suffix, candidate_multiplier in [
        ('TiB', 1024**4),
        ('GiB', 1024**3),
        ('MiB', 1024**2),
        ('KiB', 1024),
        ('TB', 1000**4),
        ('GB', 1000**3),
        ('MB', 1000**2),
        ('KB', 1000),
        ('T', 1000**4),
        ('G', 1000**3),
        ('M', 1000**2),
        ('K', 1000),
        ('B', 1),
    ]:
        if size.endswith(suffix):
            multiplier = candidate_multiplier
            size = size[:-len(suffix)]
            break
    return float(size) * multiplier

def quota_remaining(mountpoint = None):
    if mountpoint is None:
        if 'EUD_TASK_QUOTA_MOUNTPOINT' in os.environ:
            mountpoint = os.environ['EUD_TASK_QUOTA_MOUNTPOINT']
        else:
            mountpoint = '/home'
    # Following 4 lines adapted from ssoto's answer on
    # https://stackoverflow.com/questions/9781071/python-access-to-nfs-quota-information/16294121
    command = ['quota', '-f', mountpoint, '-pw']
    output = subprocess.check_output(command)
    # The 3rd line has the data
    fields = output.split(b'\n')[2].split()
    bytes_used  = 1024 * int(fields[1])
    bytes_quota = 1024 * int(fields[2])
    return bytes_quota - bytes_used

def hex2base62(h, min_length = 0):
    s = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    i = int(h, 16)
    if not i:
        return max(1, min_length) * '0'
    digits = []
    while i or (min_length and len(digits) < min_length):
        d = i % 62
        digits.append(s[d])
        i = int(i/62)
    return ''.join(digits)

def bstring(s):
    if type(b'') is str:
        return s
    if type(s) is bytes:
        return s
    return s.encode('utf-8')

def std_string(s):
    if type(b'') is str:
       return s
    if type(s) is bytes:
       return s.decode('utf-8')
    return s

def b_ord(s):
    if type(b'') is str:
        return ord(s)
    if type(s) is int:
        # probably called with binarystring[some_index]
        return s
    return s[0]   # in Python 3, accessing an element of a byte string yields an integer

def iteritems(d):
    try:
        return d.iteritems()
    except AttributeError:
        return d.items()

def makedirs(required_dir):
    if not os.path.exists(required_dir):
        try:
            os.makedirs(required_dir)
        except OSError:
            # folder was created by another process
            # between the to calls above
            # (in python 3, we will be able to use exist_ok=True)
            pass

def random_delay(max_seconds = 5.0, min_fraction = 0.0, max_fraction = 1.0):
    fraction = min_fraction * (max_fraction-min_fraction) * random()
    time.sleep(max_seconds * fraction)

def random():
    h = hashlib.sha256()
    f = open('/dev/urandom', 'rb')
    h.update(f.read(80))
    f.close()
    now = time.time()
    h.update(b'%.9f' %now)
    h = int(h.hexdigest(), 16)
    return h / 2.0**256

def get_language(lcode):
    project_dir = get_project_dir()
    retval = None
    f = open('%s/languages.tsv' %project_dir, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith('#'):
            continue
        fields = line.split()
        if fields[0] == lcode:
            retval = fields[1]
            break
    f.close()
    return retval

def get_project_dir():
    if 'PRJ_DIR' in os.environ:
        return os.environ['PRJ_DIR']
    return os.path.dirname(os.getcwd())

def get_model_dir(module_name, lcode, init_seed, datasets, options):
    if not os.path.exists(options.modeldir):
        makedirs(options.modeldir)
    h = hashlib.sha256('%s:%d:%s:%s' %(
        module_name, len(init_seed), init_seed, datasets,
    ))
    h = hex2base62(h.hexdigest(), 5)[:5]
    return '%s/%s-%s-%d-%s-%s-%s' %(
        options.modeldir, lcode, module_name, datasets.count('+') + 1,
        datasets.replace('.', '_'), init_seed, h
    )

def get_conllu_concat_filename_and_size(target_lcode, datasets, options):
    if not os.path.exists(options.tempdir):
        makedirs(options.tempdir)
    h = hashlib.sha256(datasets)
    h = hex2base62(h.hexdigest(), 5)[:5]
    filename = '%s/%s-%d-%s-%s.conllu' %(
        options.tempdir, target_lcode, datasets.count('+') + 1,
        datasets.replace('.', '_'), h,
    )
    n_tokens = 0
    if os.path.exists(filename):
        f = open(filename, 'rb')
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith('#') and b'\t' in line:
                n_tokens += 1
        f.close()
        return filename, n_tokens
    dirname, _ = filename.rsplit('/', 1)
    makedirs(dirname)
    f_out = open(filename, 'wb')
    for dataset in datasets.split('+'):
        f_out.write(b'# tbemb = %s\n' %dataset)
        # copy dataset
        conllu_file, _ = get_conllu_and_text_for_dataset(dataset, options)
        f_in = open(conllu_file, 'rb')
        while True:
            line = f_in.readline()
            if not line:
                break
            f_out.write(line)
            # count tokens
            if not line.startswith('#') and b'\t' in line:
                n_tokens += 1
        f_in.close()
    f_out.close()
    return filename, n_tokens

def get_conllu_and_text_for_dataset(dataset, options):
    dataset_type, tbid = dataset.split('.')
    if dataset_type == 'task':
        tb_info = options.task_treebanks[tbid]
    elif dataset_type == 'ud25':
        tb_info = options.ud25_treebanks[tbid]
    else:
        raise ValueError('Unknown dataset type %r in %r' %(dataset_type, dataset))
    training_conllu_path = tb_info[1]
    # https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
    training_txt_path = '.txt'.join(training_conllu_path.rsplit('.conllu', 1))
    return training_conllu_path, training_txt_path

