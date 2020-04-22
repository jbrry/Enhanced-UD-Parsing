#!/bin/bash

test -z $1 && echo "Missing allennlp version (090 or dev)"
test -z $1 && exit 1
ALLENNLP_VERSION=$1

test -z $2 && echo "Missing package version (tagging or tagging_stable)"
test -z $2 && exit 1
PACKAGE=$2

test -z $3 && echo "Missing model dir"
test -z $3 && exit 1
MODEL_DIR=$3

test -z $4 && echo "Missing input conllu file (with placeholder enhanced dependencies)"
test -z $4 && exit 1
INPUT_FILE=$4

test -z $5 && echo "Missing output conllu file"
test -z $5 && exit 1
OUTPUT_FILE=$5

if [ -e ${OUTPUT_FILE}_woc ] ; then
    echo "Re-using intermediate .connlu_woc file"
else

if [ -e $MODEL_DIR/model.tar.gz ]; then
    echo "Using $MODEL_DIR/model.tar.gz"
else
    echo "Error: $MODEL_DIR/model.tar.gz not found"
    exit 1
fi

# activate environment
source ${PRJ_DIR}/Enhanced-UD-Parsing/venv/allennlp-${ALLENNLP_VERSION}/bin/activate

# debugging
echo "sourced ${PRJ_DIR}/Enhanced-UD-Parsing/venv/allennlp-${ALLENNLP_VERSION}/bin/activate"
echo "using package $PACKAGE"

allennlp predict  \
    ${MODEL_DIR}/model.tar.gz        \
    ${INPUT_FILE}                      \
    --output-file ${OUTPUT_FILE}_woc  \
    --predictor enhanced-predictor     \
    --include-package "$PACKAGE"       \
    --use-dataset-reader               \
    --silent
fi
  
if [ -e ${OUTPUT_FILE}_woc ]; then
    if scripts/restore-conllu-comments-and-more.py ${INPUT_FILE} < ${OUTPUT_FILE}_woc > ${OUTPUT_FILE}_ra ; then
        mv ${OUTPUT_FILE}_ra ${OUTPUT_FILE}
    else
        # failure
        echo "Error: Helper script return with non-zero error code. Please inspect:"
        echo "Parser input: ${INPUT_FILE}"
        echo "Parser output: ${OUTPUT_FILE}_woc"
        echo "Restored output: ${OUTPUT_FILE}_ra"
    fi
    # delete _woc file and, if present, _ra file
    rm ${OUTPUT_FILE}_*
else
    echo "Error: No output file"
fi

