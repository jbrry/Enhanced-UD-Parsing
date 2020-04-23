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
import random
import subprocess
import sys

import conllu_dataset

def uses_external_models():
    return True

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
    parts = []
    parts.append('copy2e')
    for part in __name__.split('_')[2:]:
        parts.append(part)
    return '_'.join(parts)

def predict(lcode, init_seed, dataset, options, conllu_input_file, conllu_output_file):
    if __name__.count('_') == 1:
        # plain copy
        copy_basic_to_enhanced(conllu_input_file, conllu_output_file)
        return True
    # create a temporary copy with enhanced dependencies copied from
    # the basic tree
    # TODO: we assume here the input is in our temp folder but
    #       for general use we should make sure to use our temp
    #       folder even when the input comes from somewhere else
    conllu_input_copy2enh = conllu_input + '_c2e'
    if not os.path.exists(conllu_input_copy2enh):
        copy_basic_to_enhanced(
            conllu_input, conllu_input_copy2enh
        )
    # now read this file sentence-by-sentence
    apply_heuristic_rules(conllu_input_copy2enh, conllu_output_file)
    # cleanup _c2e file
    if not options.debug:
        os.unlink(conllu_input_copy2enh)
    # TODO: check output more carefully,
    #       e.g. check number of sentences (=number of empty lines)
    return os.path.exists(conllu_output)

def apply_heuristic_rules(conllu_input_file, conllu_output_file):
    f_in = open(conllu_input_file, 'rb')
    f_out = open(conllu_output_file, 'wb')
    raise NotImplementedError
    while True:
        line = f_in.readline()
        if not line:
            break
        sentence = conllu_dataset.ConlluSentence()
        sentence.append(line)
        while True:
            line = f_in.readline()
            if line.isspace():
                # empty line --> end of sentence
                break
            sentence.append(line)
        # sentence is ready
        candidates = {}  # head --> list of new canidate labels
        if '_encase' in __name__:
            collect_en_case_candidates(sentence, candidates)
        if '_arcase' in __name__:
            collect_ar_case_candidates(sentence, candidates)
        if '_mark' in __name__:
            collect_mark_candidates(sentence, candidates)
        if '_cc' in __name__:
            collect_cc_candidates(sentence, candidates)
        apply_candidate(sentence, candidates)
        if '_rel' in __name__:
            apply_rel_rule(sentence)
        # write result
        sentence.write(f_out)
    f_in.close()
    f_out.close()

def collect_en_case_candidates(sentence, candidates):
    '''
        For each row with label "case", find its head and if the head's
        label is not "obl:agent" append the row's lemma to the label of
        the head's arc also attested in the basic tree.
    '''
    for row_index in sentence.enh_tokens2row:
        row = sentence.rows[row_index]             # y
        label = row[conllu_dataset.label_column]   # label(y)
        if label != 'case':
            break
        head = row[conllu_dataset.head_column]     # head(y)
        head_row_index = sentence.id2row[head]
        head_row = sentence.rows[head_row_index]   # x
        head_label = head_row[conllu_dataset.label_column]   # label(x)
        if head_label == 'obl:agent':
            continue
        if not row_index in candidates:
            candidates[row_index] = []
        candidates[head_row_index].append('%s:%s:%s' %(
            head_row[conllu_dataset.head_column],  # head(x)
            head_label,                            # label(x)
            row[conllu_dataset.lemma_column],      # lemma(y)
        ))

def collect_ar_case_candidates(sentence, candidates):
    '''
        Like en-case but no exclusion of "obl:agent" and, in addition to
        the lemma, append lowercase version of value of case attribute
        in morphological features.
    '''
    for row_index in sentence.enh_tokens2row:
        row = sentence.rows[row_index]             # y
        label = row[conllu_dataset.label_column]   # label(y)
        if label != 'case':
            break
        head = row[conllu_dataset.head_column]     # head(y)
        head_row_index = sentence.id2row[head]
        head_row = sentence.rows[head_row_index]   # x
        head_label = head_row[conllu_dataset.label_column]   # label(x)
        #if head_label == 'obl:agent':
        #    continue
        morph = head_row[conllu_dataset.morph_column].lower()
        case_value = None
        for feature in morph.split('|'):
            if feature.startswith('case='):
                _, case_value = feature.split('=', 1)   # case_value(x)
                if ',' in case_value:
                     # must disambiguate case
                     # TODO: use classifier
                     print('Warning: Making random choice among case values %s' %case_value)
                     values = case_value.split(',')
                     case_value = random.choice(values)
        if not case_value:
            print('Warning: Found no case value. Using gen')
            case_value = 'gen'
        if not row_index in candidates:
            candidates[row_index] = []
        candidates[head_row_index].append('%s:%s:%s:%s' %(
            head_row[conllu_dataset.head_column],  # head(x)
            head_label,                            # label(x)
            row[conllu_dataset.lemma_column],      # lemma(y)
            case_value
        ))

def collect_mark_candidates(sentence, candidates):
    '''
        For each row with label "mark", find its head and if the head's
        label is "acl" or "advcl" append the row's lemma to the label
        of the head's arc also attested in the basic tree.
    '''
    for row_index in sentence.enh_tokens2row:
        row = sentence.rows[row_index]             # y
        label = row[conllu_dataset.label_column]   # label(y)
        if label != 'mark':
            break
        head = row[conllu_dataset.head_column]     # head(y)
        head_row_index = sentence.id2row[head]
        head_row = sentence.rows[head_row_index]   # x
        head_label = head_row[conllu_dataset.label_column]   # label(x)
        if head_label not in ('acl', 'advcl'):
            continue
        if not row_index in candidates:
            candidates[row_index] = []
        candidates[head_row_index].append('%s:%s:%s' %(
            head_row[conllu_dataset.head_column],  # head(x)
            head_label,                            # label(x)
            row[conllu_dataset.lemma_column],      # lemma(y)
        ))

def collect_cc_candidates(sentence, candidates):
    '''
        For each row with label "cc", find its head and append the row's
        lemma to the label of the head's arc also attested in the basic
        tree.
    '''
    for row_index in sentence.enh_tokens2row:
        row = sentence.rows[row_index]             # y
        label = row[conllu_dataset.label_column]   # label(y)
        if label != 'cc':
            break
        head = row[conllu_dataset.head_column]     # head(y)
        head_row_index = sentence.id2row[head]
        head_row = sentence.rows[head_row_index]   # x
        head_label = head_row[conllu_dataset.label_column]   # label(x)
        if not row_index in candidates:
            candidates[row_index] = []
        candidates[head_row_index].append('%s:%s:%s' %(
            head_row[conllu_dataset.head_column],  # head(x)
            head_label,                            # label(x)
            row[conllu_dataset.lemma_column],      # lemma(y)
        ))

def apply_candidates(sentence, candidates, ignore_existing = True):
    for row_index in candidates:
        row = sentence.rows[row_index]
        enh_dep_rel_set = candidates[row_index]
        if not ignore_existing:
            # include existing relations in set to be merged
            enh = row[conllu_dataset.enh_column]
            for enh_dep_rel in enh.split('|'):
                enh_dep_rel_set.append(enh_dep_rel)
        # allocate separate buckets for each head
        head2bucket = {}
        for enh_dep_rel in enh_dep_rel_set:
            head, labels = enh_dep_rel.split(':', 1)
            if not head in head2bucket:
                head2bucket[head] = set()
            # delete any prefix of this label
            to_be_deleted = []
            for label_in_bucket in head2bucket[head]:
                if label.startwith(label_in_bucket)  \
                and label != label_in_bucket:
                    to_be_deleted.append(label_in_bucket)
            for label_in_bucket in to_be_deleted:
                head2bucket[head].remove(label_in_bucket)
            # add this label
            head2bucket[head].add(labels)
        # sort by float(head)
        heads_float_and_key = []
        for head in head2bucket:
            try:
                fl_head = float(head)
            except:
                print('Warning: Cannot interpret head %r as float' %head)
                fl_head = 9999.9
            heads_float_and_key.append((fl_head, head))
        heads_float_and_key.sort()
        # assemble new list of arcs
        new_arcs = []
        for _, head in heads_float_and_key:
            bucket = head2bucket[head]
            if len(bucket) == 1:
                new_arcs.append('%s:%s' %(head, bucket.pop()))
            elif not bucket:
                # should not happen
                print('Warning: Missing label in bucket. Using a frequent one')
                new_arcs.append('%s:nmod' %head)
            else:
                # pick the longest candidate label
                # TODO: use a classifier to decide
                clabels = []
                for label in bucket:
                    clabels.append((-label.count(':'), -len(label), label))
                clabels.sort()
                _, _, label = clabels[0]
                new_arcs.append('%s:%s' %(head, label))
        # write list of arcs into enhanced dependencies column
        row[conllu_dataset.enh_column] = '|'.join(new_arcs)

def apply_rel_rule(sentence):
    raise NotImplementedError

def copy_basic_to_enhanced(conllu_input_file, conllu_output_file):
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

