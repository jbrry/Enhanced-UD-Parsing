#!/bin/bash

PRED_DIR=test-pred-2020-04-25

MAX_JOBS=8

for CONLLU in $(ls $PRED_DIR/ | fgrep .conllu | LC_ALL=C sort) ; do
    echo "== $CONLLU =="
    for QUICKFIX in dans-quick-fix-*.pl ; do
        NAME=$(basename $QUICKFIX .pl)
        mkdir -p $PRED_DIR/$NAME
        cat $PRED_DIR/$CONLLU | \
            cut -f-10         | \
            perl $QUICKFIX --connect-to-root | \
            tee $PRED_DIR/$NAME/$CONLLU > /dev/null &
        # limit number of tasks running in parallel
        while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
            sleep 1
        done
    done
done

wait

