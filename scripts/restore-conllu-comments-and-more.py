#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# usage: restore-conllu-comments-and-more.py ${CONLLU_BEFORE_PARSING} < ${CONLLU_AFTER_PARSING} > ${GOOD_OUTPUT}

import os
import time
import sys

def main():
    file_with_comments = open(sys.argv[1], 'rb')
    have_warned_about_trailing_output = False
    line2 = None
    while True:
        line1 = file_with_comments.readline()
        if line2 is None:
            line2 = sys.stdin.readline()
        if not line1:
            if line2:
                if not have_warned_about_trailing_output:
                    sys.stderr.write('Warning: unexpected trailing enhancer output\n')
                    have_warned_about_trailing_output = True
                sys.stdout.write(line2)
                line2 = None
                continue
            break
        id1 = line1.split('\t')[0]
        id2 = line2.split('\t')[0]
        if id1 == id2:
            # lines are in sync --> use more complete annotation
            sys.stdout.write(line2)
            line2 = None
            continue
        if line1.startswith('#'):
            #if line2.startswith('#'):
                # both have comments but they do not match
                # --> let's assume the parser never adds new
                # comments --> restore the missing comment
                # (lines should be in sync soon again)
                #sys.stdout.write(line1)
                #continue
            # comment line is missing
            sys.stdout.write(line1)
            continue
        if line2.startswith('#'):
            raise ValueError('Not supported: parser added a comment')
        if '-' in id1 and id1.split('-')[0] == id2:
            # x-y token is missing
            sys.stdout.write(line1)
            continue
        if '.' in id1 and id1.split('.')[0] == id2:
            # x.y token is missing
            sys.stdout.write(line1)
            continue
        if '.' in id2 and id2.split('.')[0] == id1:
            # x.y token was added
            sys.stdout.write(line2)
            continue
        raise ValueError('Do not know how to handle line1 = %r with line2 = %r' %(line1, line2))
    file_with_comments.close()
        
if __name__ == "__main__":
    main()

