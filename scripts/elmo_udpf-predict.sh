#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing model folder"
test -z $1 && exit 1
MODELDIR=$1

test -z $2 && echo "Missing conllu input file"
test -z $2 && exit 1
INPUT=$2

test -z $3 && echo "Missing npz input file"
test -z $3 && exit 1
IN_NPZ=$3

test -z $4 && echo "Missing conllu output file"
test -z $4 && exit 1
OUTPUT=$4

test -z $5 && echo "Missing language code"
test -z $5 && exit 1
LANG_CODE=$5

EXTRA_OPTIONS="$6"

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/enhanced-ud/Enhanced-UD-Parsing

WORKDIR=$(realpath "${OUTPUT}")-workdir
mkdir -p "${WORKDIR}"

LANG_CODE=$(cat ${MODELDIR}/elmo-lcode.txt)

ELMO_FILE_PREFIX=elmo

ln -s $(realpath "${IN_NPZ}") ${WORKDIR}/${ELMO_FILE_PREFIX}-test.npz

#source ${PRJ_DIR}/config/locations.sh

PARSER_NAME=udpipe-future
PARSER_DIR=${UDPIPE_FUTURE_DIR}

if [ -n "$UDPIPE_FUTURE_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${UDPIPE_FUTURE_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$UDPIPE_FUTURE_ICHEC_CONDA" ]; then
    module load conda/2
    source activate ${UDPIPE_FUTURE_ICHEC_CONDA}
fi

if [ -n "$UDPIPE_FUTURE_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${UDPIPE_FUTURE_CONDA}
fi

if [ -n "$UDPIPE_FUTURE_ENV" ]; then
    source ${UDPIPE_FUTURE_ENV}/bin/activate
fi

FAKE_TBID=xx_xxx

REAL_MODELDIR=$(realpath $MODELDIR)
REAL_INPUT=$(realpath $INPUT)
REAL_OUTPUT=$(realpath $OUTPUT)

cd ${WORKDIR}

for ENTRY in   \
    checkpoint  \
    checkpoint-inference-last.data-00000-of-00001  \
    checkpoint-inference-last.index                \
    fasttext.npz                                   \
    xx_xxx-ud-train.conllu                         \
; do
    ln -s ${REAL_MODELDIR}/${ENTRY}
done

python ${PARSER_DIR}/ud_parser.py  \
    ${EXTRA_OPTIONS}                \
    --predict                        \
    --predict_input "${REAL_INPUT}"    \
    --predict_output "${REAL_OUTPUT}"   \
    --elmo ${ELMO_FILE_PREFIX}-test.npz  \
    --embeddings fasttext.npz             \
    --logdir ./                            \
    --checkpoint checkpoint-inference-last  \
    ${FAKE_TBID}                            \
    2> "${REAL_OUTPUT}"-stderr.txt                           \
    > /dev/null

cd /
rm -rf ${WORKDIR}

if [ -e "${REAL_OUTPUT}" ]; then
    rm "${REAL_OUTPUT}"-stderr.txt
fi

if [ -n "$UDPIPE_FUTURE_DELETE_INPUT_NPZ" ]; then
   rm -f "$IN_NPZ"
fi

