#!/bin/bash

echo "== (1) Using Dan's connect-to-root =="

PRED_DIR=effect-of-connect-to-root
MAX_JOBS=8

QUICKFIX=dans-quick-fix-1d3be2b.pl
for CONLLU in $(ls $PRED_DIR/ | fgrep .conllu ) ; do
    echo "== $CONLLU =="
    NAME=$(basename $QUICKFIX .pl)
    mkdir -p $PRED_DIR/$NAME
    cat $PRED_DIR/$CONLLU | \
        cut -f-10         | \
        perl $QUICKFIX --connect-to-root   | \
        tee $PRED_DIR/$NAME/$CONLLU > /dev/null &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 1
    done
done

echo "== (2) Using our own connect-graph.py =="

MAX_JOBS=8
OURFIX=Enhanced-UD-Parsing/scripts/wrapper-connect-graph.sh

QUICKFIX=dans-quick-fix-1d3be2b.pl
for CONLLU in $(ls $PRED_DIR/ | fgrep .conllu | LC_ALL=C sort) ; do
    echo "== $CONLLU =="
    NAME=n04-fragment-fix-and-1d3be2b
    mkdir -p $PRED_DIR/$NAME
    cat $PRED_DIR/$CONLLU | \
        cut -f-10         | \
        $OURFIX 2> $PRED_DIR/$NAME/${CONLLU}.err | \
        perl $QUICKFIX    | \
        tee $PRED_DIR/$NAME/$CONLLU > /dev/null &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 1
    done
done

wait

