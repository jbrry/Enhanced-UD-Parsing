#!/bin/bash

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner


#SBATCH -p compute       # which partition to run on
#SBATCH -J conll17   # name for the job
#SBATCH -d singleton
#SBATCH --mem=128900
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH -N 1-1
#SBATCH --constraint=faststorage

# For best learning, data needs to be shuffled so that batches are not biased.
# Looking at fasttext code, it seems to work line by line as newlines
# are translated to the special word `EOS = '</s>'`.
# --> one sentence per line format


CONLL17_DIR=${HOME}/data/2-conll17
TMP_DIR=${CONLL17_DIR}/tmp
test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/enhanced-ud/Enhanced-UD-Parsing

SCRIPT_DIR=${PRJ_DIR}/scripts
OUTPUT_DIR=text

MAX_JOBS=60

cd ${CONLL17_DIR}
mkdir -p $TMP_DIR
mkdir -p ${OUTPUT_DIR}

for LTAR in *-annotated-conll17.tar ; do
    L=$(basename ${LTAR} -annotated-conll17.tar)
    echo "== $L started $(date) =="
    tar -xf ${LTAR}
    echo "$L extracted $(date)"
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
    echo "$L jobs finished $(date)"
    rm -rf ${L}
    echo "$L xz data deleted $(date)"
    echo "Merging data for ${L}..."
    LC_ALL=C sort --merge --batch-size=5 -S 80M ${TMP_DIR}/$$-$L-*.tsv | \
        cut -f2- > ${OUTPUT_DIR}/$L.txt
    echo "$L merged $(date)"
    echo "Cleaning up..."
    rm -f ${TMP_DIR}/$$-$L-*.tsv
done
echo "== finished $(date) =="

