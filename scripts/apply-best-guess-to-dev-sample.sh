#!/bin/bash

if [ -n $UD_TOOLS_DIR ] ; then
    echo "Detected UD tools dir"
else
    echo "Error: UD_TOOLS_DIR not set"
    exit 1
fi

if [ -n $(fgrep best_guess scripts/wrapper-collapse-empty-nodes.sh) ] ; then
    echo "Detected best_guess in connect-graph.py - hope it isn't a comment"
else
    echo "Error: no --mode best_guess in connect-graph.py call"
fi

PRED_DIR=effect-of-connect-to-root
MAX_JOBS=8
OURFIX=Enhanced-UD-Parsing/scripts/wrapper-connect-graph.sh

QUICKFIX=dans-quick-fix-1d3be2b.pl
for CONLLU in $(ls $PRED_DIR/ | fgrep .conllu | LC_ALL=C sort) ; do
    echo "== $CONLLU =="
    NAME=best-guess-fragment-fix-and-1d3be2b
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

