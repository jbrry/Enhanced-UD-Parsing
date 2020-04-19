#!/bin/bash

test -z $1 && echo "Missing task type <basic> or <enhanced>"
test -z $1 && exit 1
TASK=$1

test -z $2 && echo "Missing model type <dm> or <kg>"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing list of TBIDs (space or colon-separated)"
test -z $3 && exit 1
TBIDS=$(echo $3 | tr ':' ' ')

test -z $4 && echo "Missing language code for BERT; currently supports: (ar bg_cs_pl_ru en fi fr it mbert nl pl ru sv)"
test -z $4 && exit 1
BERT_MODEL=$4

test -z $5 && echo "Missing random seed"
test -z $5 && exit 1
RANDOM_SEED=$5

# official shared-task data
#TB_DIR=data/train-dev
TB_DIR=data/train-dev-filtered

TRANSFORMER_DIR=${HOME}/transformer_dir/pytorch_models

TIMESTAMP=`date "+%Y%m%d-%H%M%S"` 

N_SHORT=`echo ${HOSTNAME} | cut -c-5 `
if [ "${N_SHORT}" = "node0" ]; then
  echo "loading CUDA"
  module add cuda10.1
fi

for tbid in $TBIDS ; do
  echo
  echo "== $tbid =="
  echo

  # seeds
  SEED=$RANDOM_SEED
  PYTORCH_SEED=`expr $SEED / 10`
  NUMPY_SEED=`expr $PYTORCH_SEED / 10`
  export RANDOM_SEED=$SEED
  export PYTORCH_SEED=$PYTORCH_SEED
  export NUMPY_SEED=$NUMPY_SEED

  # hyperparams
  export BATCH_SIZE=16
  export NUM_EPOCHS=75
  export CUDA_DEVICE=0
 
  #if [ "$tbid" = "fr_sequoia" ] || [ "$tbid" = "ru_syntagrus" ]; then
    # lemmas, upos, feats
  #  FEATS=luf
  #else
    # lemmas, upos, xpos, feats
  #  FEATS=luxf
  #fi
  
  FEATS=u
  CONFIG=configs/stable/ud_${TASK}_bert_${MODEL_TYPE}_${FEATS}.jsonnet

  if [ "$tbid" = "et_ewt" ]; then
    CONFIG=configs/stable/ud_${TASK}_bert_${MODEL_TYPE}_${FEATS}_no_dev.jsonnet
  fi


  for filepath in ${TB_DIR}/*/${tbid}-ud-train.conllu; do
    dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
    tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

    # ud v2.x
    export TRAIN_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-train.conllu
    export DEV_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-dev.conllu
    export TEST_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-test.conllu

    # BERT params
    export BERT_VOCAB=${TRANSFORMER_DIR}/${BERT_MODEL}/vocab.txt
    export BERT_WEIGHTS=${TRANSFORMER_DIR}/${BERT_MODEL}/weights.tar.gz
  
    allennlp train ${CONFIG} -s logs_stable/${tbid}-${TASK}-${MODEL_TYPE}-${BERT_MODEL}-BERT-${FEATS}-${TIMESTAMP} --include-package tagging_stable
  done
done

