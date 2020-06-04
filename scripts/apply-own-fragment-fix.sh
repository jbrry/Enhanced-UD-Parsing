#!/bin/bash

PRED_DIR=test-pred-2020-04-25/identified-basis-for-1259
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

