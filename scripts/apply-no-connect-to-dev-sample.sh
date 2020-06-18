#!/bin/bash

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner


echo "== (1) Using Dan's connect-to-root =="

PRED_DIR=effect-of-connect-to-root
MAX_JOBS=8

QUICKFIX=dans-quick-fix-1d3be2b.pl
for CONLLU in $(ls $PRED_DIR/ | fgrep .conllu ) ; do
    echo "== $CONLLU =="
    NAME=dans-without-connect-to-root
    mkdir -p $PRED_DIR/$NAME
    cat $PRED_DIR/$CONLLU | \
        cut -f-10         | \
        perl $QUICKFIX    | \
        tee $PRED_DIR/$NAME/$CONLLU > /dev/null &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 1
    done
done

wait

