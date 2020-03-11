#!/usr/bin/env bash

test -z $1 && echo "Missing task type <basic> or <enhanced>"
test -z $1 && exit 1
TASK=$1

test -z $2 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $2 && exit 1
TBIDS=$(echo $2 | tr ':' ' ')

# TODO: iterate over TBIDS and BERT_TYPES simultaneously so this script can run with multiple TBIDS and corresponding BERT models.
test -z $3 && echo "Missing language code for BERT; currently supports: (ar bg-cs-pl-ru en fi it mbert nl pl sv)"
test -z $3 && exit 1
BERT_TYPE=$3

# official shared-task data
TB_DIR=data/train-dev
TRANSFORMER_DIR=${HOME}/transformer_dir/pytorch_models

TIMESTAMP=`date "+%Y%m%d-%H%M%S"` 

for tbid in $TBIDS ; do
  echo
  echo "== $tbid =="
  echo

  for filepath in ${TB_DIR}/*/${tbid}-ud-train.conllu; do
  dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
  tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

  # ud v2.x
  export TRAIN_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-train.conllu
  export DEV_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-dev.conllu
  export TEST_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-test.conllu

  # BERT params
  export BERT_VOCAB=${TRANSFORMER_DIR}/${BERT_TYPE}/vocab.txt
  export BERT_WEIGHTS=${TRANSFORMER_DIR}/${BERT_TYPE}/weights.tar.gz
  
  allennlp train configs/ud_${TASK}_bert.jsonnet -s logs/${tbid}-${TASK}-${BERT_TYPE}-BERT-${TIMESTAMP} --include-package tagging
  done
done

