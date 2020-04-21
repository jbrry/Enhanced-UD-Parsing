#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import hashlib
import os
import random as py_random
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
    if 'udpf' in module_name.split('_'):
        return get_udpf_model_dir(module_name, lcode, init_seed, datasets, options)
    elif module_name.startswith('allennlp_'):
        return get_allennlp_model_dir(module_name, lcode, init_seed, datasets, options)
    else:
        raise ValueError('get_model_dir() with module_name %s' %module_name)

def get_allennlp_model_dir(module_name, lcode, init_seed, datasets, options):
    module_fields = module_name.split('_')
    # example: [0] [1] [2] [3]  [4]
    #     allennlp_090_dm_mbert_luxf
    allennlp_modeldir = '%s/allennlp-%s' %(options.modeldir, module_fields[1])
    candidates = []
    now = time.time()
    if options.debug:
        print('Scanning %s for %s with %s and %s' %(
            allennlp_modeldir, module_name, lcode, datasets
        ))
    for candidate_model in sorted(os.listdir(allennlp_modeldir)):
        fields = candidate_model.split('-')
        # example: [0] [1]    [2] [3] [4]  [5] [6]      [7]
        #      en_ewt-enhanced-dm-en-BERT-luxf-20200417-060431
        reject_reason = None
        if not fields[0].startswith(lcode):
            reject_reason = 'wrong lcode'
        if fields[1] != 'enhanced':
            reject_reason = 'not for enhanced parsing'
        if fields[2] != module_fields[2]:
            reject_reason = 'dm/kg mismatch'
        if fields[5] != module_fields[4]:
            reject_reason = 'not the right input features'
        if module_fields[3] == 'mbert':
            if fields[3] != 'mbert':
                reject_reason = 'not an mbert model'
        elif module_fields[3] == 'pbert':
            # polyglotly fine-tuned mbert
            if not '_' in fields[3]:
                reject_reason = 'list of bert languages too short'
        elif module_fields[3] == 'lbert':
            # language-specific bert
            if fields[3] != lcode:
                print('Warning: Found allennlp model for %s using BERT for %s' %(lcode, fields[3]))
        else:
            raise ValueError('Unsupported allennlp BERT type')
        model_file = '%s/%s/model.tar.gz' %(allennlp_modeldir, candidate_model)
        if not os.path.exists(model_file):
            reject_reason = 'missing model.tar.gz'
        if not reject_reason:
            model_tbid = fields[0]
            priority = 1
            details = 'tbid mismatch'
            model_id = '%s_%s' %(module_name, lcode)
            for dataset in datasets.split('+'):
                dataset_tbid = dataset.split('.')[-1]
                if model_tbid == dataset_tbid:
                    priority = 0
                    details = 'tbid match'
                    model_id = '%s_%s' %(module_name, model_tbid)
                    break
            model_age = now - os.path.getmtime(model_file)
            candidates.append((priority, model_age, candidate_model, model_id))
            if options.debug:
                print('\t%s\t-->\tOK (%s)' %(candidate_model, details))
        elif options.debug:
            print('\t%s\t-->\t%s' %(candidate_model, reject_reason))
    if not candidates:
        return None, 'none'
    candidates.sort()
    _, _, model_dir, model_id = candidates[0]
    return model_dir, model_id

def get_udpf_model_dir(module_name, lcode, init_seed, datasets, options):
    udpf_modeldir = '/'.join(options.modeldir, 'udpf')
    makedirs(udpf_modeldir)
    h = hashlib.sha256('%s:%d:%s:%s' %(
        module_name, len(init_seed), init_seed, datasets,
    ))
    h = hex2base62(h.hexdigest(), 5)[:5]
    model_id = '%s-%d-%s-%s' %(
        module_name, datasets.count('+') + 1,
        datasets.replace('.', '_'), init_seed
    )
    model_path = '%s/%s-%s-%s' %(
        udpf_modeldir, lcode,
        model_id,
        h
    )
    return model_path, model_id

def get_conllu_size(filename):
    n_tokens = 0
    f = open(filename, 'rb')
    while True:
        line = f.readline()
        if not line:
            break
        if not line.startswith('#') and b'\t' in line:
            n_tokens += 1
    f.close()
    return n_tokens

def get_conllu_concat_filename_and_size(
    target_lcode, datasets, options,
    dataset_partition,
):
    if not os.path.exists(options.tempdir):
        makedirs(options.tempdir)
    h = hashlib.sha256(datasets)
    h = hex2base62(h.hexdigest(), 5)[:5]
    filename = '%s/%s-%d-%s-%s-%s.conllu' %(
        options.tempdir, target_lcode, datasets.count('+') + 1,
        dataset_partition,
        datasets.replace('.', '_'), h,
    )
    if os.path.exists(filename):
        if options.debug:
            print('Re-using', filename)
        return filename, get_conllu_size(filename)
    datasets_with_conllu = []
    for dataset in datasets.split('+'):
        conllu_file, _ = get_conllu_and_text_for_dataset(
            dataset, options, dataset_partition
        )
        if conllu_file:
            datasets_with_conllu.append((dataset, conllu_file))
        else:
            print('Warning: Partition %s in %s not found for %s' %(
                dataset_partition, dataset, datasets
            ))
    if not datasets_with_conllu:
        # nothing to concatenate; refuse to create empty dataset
        return None, 0
    dirname, _ = filename.rsplit('/', 1)
    makedirs(dirname)
    if dataset_partition == 'train' and '+' not in datasets:
        # no need to copy training files for mono-treebank models
        # (dev files with just one tbid may still need tbemb annotation
        # in case they are used to evaluate a multi-treebank model)
        link_target = os.path.abspath(datasets_with_conllu[0][1])
        os.symlink(link_target, filename)
        return filename, get_conllu_size(filename)
    n_tokens = write_multi_treebank_conllu(filename, datasets_with_conllu)
    return filename, n_tokens

def get_conllu_subset(input_conllu_path, fraction, options):
    subset_dir = '%s/subsets' %options.tempdir
    makedirs(subset_dir)
    _, basename = input_conllu_path.rsplit('/', 1)
    if basename.endswith('.conllu'):
        basename = basename[:-7]
    frac_str = '%.6f' %fraction
    subset_conllu_path = '%s/%s-%s.conllu' %(
        subset_dir, basename, frac_str
    )
    n_tokens = write_multi_treebank_conllu(
        subset_conllu_path,
        [(None, input_conllu_path)],   # None = do not add tbemb tag
        fraction = fraction,
    )
    if not n_tokens:
        raise ValueError('Created empty subset even though first sentence should be included. Empty input %s?' %input_conllu_path)
    return subset_conllu_path

def write_multi_treebank_conllu(
    filename, datasets_with_conllu,
    random_choices = None,
    fraction = 1.0,
):
    f_out = open(filename, 'wb')
    n_tokens_read = 0.0
    n_tokens_copied = 0
    copy_this_sentence = True
    for dataset, conllu_file in datasets_with_conllu:
        # copy dataset
        f_in = open(conllu_file, 'rb')
        start_of_sentence = True
        while True:
            line = f_in.readline()
            if not line:
                break
            if start_of_sentence and fraction < 1.0 and n_tokens_read > 0:
                copy_this_sentence = n_tokens_copied / n_tokens_read < fraction
            if start_of_sentence and copy_this_sentence:
                if dataset is not None:
                    f_out.write(b'# tbemb = %s\n' %dataset)
                elif random_choices:
                    f_out.write(b'# tbemb = %s\n' %py_random.choice(random_choices))
            if copy_this_sentence:
                f_out.write(line)
            # count tokens
            if not line.startswith('#') and b'\t' in line:
                n_tokens_read += 1.0
                if copy_this_sentence:
                    n_tokens_copied += 1
            # Is next line a new sentence?
            # Yes (true) if current line is empty
            start_of_sentence = not line.rstrip()
        f_in.close()
    f_out.close()
    return n_tokens_copied

def conllu_with_tbemb(datasets, options, conllu_input, proxy_tbid):
    tbemb = None
    all_tbemb = []
    for dataset in sorted(datasets.split('+')):
        if dataset.endswith(proxy_tbid):
            tbemb = dataset
            break
        all_tbemb.append(dataset)
    tbembdir = '%s/with-proxy' %options.tempdir
    makedirs(tbembdir)
    _, basename = conllu_input.rsplit('/', 1)
    if basename.endswith('.conllu'):
        basename = basename[:-7]
    if tbemb:
        filename = '%s/%s-proxy_%s.conllu' %(
            tbembdir, basename, tbemb.replace('.', '_')
        )
        write_multi_treebank_conllu(filename, [(tbemb, conllu_input),])
    else:
        all_tbemb.sort()
        filename = '%s/%s-proxy_random_%s.conllu' %(
            tbembdir, basename, '_'.join(all_tbemb).replace('.', '_')
        )
        write_multi_treebank_conllu(
            filename, [(None, conllu_input),],
            random_choices = all_tbemb
        )
    return filename

def get_conllu_and_text_for_dataset(dataset, options, dataset_partition = 'train'):
    dataset_type, tbid = dataset.split('.')
    if dataset_type == 'task':
        tb_info = options.task_treebanks[tbid]
    elif dataset_type == 'ud25':
        tb_info = options.ud25_treebanks[tbid]
    else:
        raise ValueError('Unknown dataset type %r in %r' %(dataset_type, dataset))
    if dataset_partition == 'dev':
        dev_conllu = tb_info[4]
        if dev_conllu is None:
            return None, None
        # https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
        dev_txt = '.txt'.join(dev_conllu.rsplit('.conllu', 1))
        return dev_conllu, dev_txt
    if dataset_partition != 'train':
        raise ValueError('Cannot use conllu and text files for partition %s in shared task' %dataset_partition)
    training_conllu_path = tb_info[1]
    # https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
    training_txt_path = '.txt'.join(training_conllu_path.rsplit('.conllu', 1))
    return training_conllu_path, training_txt_path

def get_training_details(lcode, init_seed, datasets, options, module_name, max_tr_tokens = 60000000):
    assert '.' in datasets
    model_dir = get_model_dir(
        module_name, lcode, init_seed, datasets, options,
    )[0]
    if os.path.exists(model_dir):
        if options.debug:
            print('Not providing training details for model that already exists: %s' %model_dir)
        return None, None, None, None
    if model_dir in options.in_progress:
        if options.debug:
            print('Not providing training details for model that is in progress being trained: %s' %model_dir)
        return None, None, None, None
    tr_data_filename, n_tokens = get_conllu_concat_filename_and_size(
        lcode, datasets, options, 'train',
    )
    if not tr_data_filename:
        raise ValueError('Could not find conllu for %s' %datasets)
    monitoring_datasets = []
    for test_partition in ('dev',):  # testing on 'test' not allowed in shared task
        test_data_path, _ = get_conllu_concat_filename_and_size(
            lcode, datasets, options, test_partition,
        )
        if test_data_path:
            monitoring_datasets.append(test_data_path)
    if not monitoring_datasets:
        # use 5% of training data to satisfy udpf's requirement
        # for a test set --> scores will no be meaningful
        test_data_path = get_conllu_subset(tr_data_filename, 0.05, options)
    epochs = 80
    while epochs * n_tokens > max_tr_tokens and epochs > 6:
        epochs -= 1
    options.in_progress.add(model_dir)
    return tr_data_filename, monitoring_datasets, model_dir, epochs

def wait_for_tasks(task_list):
    for task in task_list:
        if task is None:
            continue
        task.wait()
        try:
            cleanup = task.cleanup_object
        except:
            cleanup = None
        if cleanup is not None:
            cleanup.cleanup()
