#!/bin/bash

# For best learning, data needs to be shuffled so that batches are not biased.
# Looking at fasttext code, it seems to work line by line as newlines
# are translated to the special word `EOS = '</s>'`.
# --> one sentence per line format

TMP_DIR=/tmp

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/enhanced-ud/Enhanced-UD-Parsing

SCRIPT_DIR=${PRJ_DIR}/scripts

OUTPUT_DIR=text

mkdir -p ${OUTPUT_DIR}

MAX_JOBS=12

for L in \
    Uyghur \
    Irish \
    Hungarian \
    English \
    Vietnamese \
; do
    echo "== $L started $(date) =="
    for I in $L/*.conllu.xz ; do
        TMP=${TMP_DIR}/$$-$L-$(basename $I .conllu.xz).tsv
        unxz < $I | \
            ${SCRIPT_DIR}/get-conllu-text.py \
            --info $I  \
	    --random-prefix | \
            LC_ALL=C sort -S 1G > ${TMP} &
        # limit number of tasks running in parallel
        while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
            sleep 5
        done
    done
    echo "Waiting for $L jobs to finish..."
    wait
    echo "Merging data for ${L}..."
    LC_ALL=C sort --merge --batch-size=5 -S 80M ${TMP_DIR}/$$-$L-*.tsv | \
        cut -f2- > ${OUTPUT_DIR}/$L.txt
    echo "Cleaning up..."
    rm -f ${TMP_DIR}/$$-$L-*.tsv
done
echo "== finished $(date) =="

