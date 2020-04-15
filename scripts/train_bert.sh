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
TB_DIR=data/train-dev

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
  export BATCH_SIZE=8
  export NUM_EPOCHS=75
  export CUDA_DEVICE=0
  export GRAD_ACCUM_BATCH_SIZE=32

  if [ "$tbid" = "fr_sequoia" ] || [ "$tbid" = "ru_syntagrus" ]; then
    # lemmas, upos, feats
    FEATS=luf
  else
    # lemmas, upos, xpos, feats
    FEATS=luxf
  fi

  # get user and model names on huggingface
  if [ "${BERT_MODEL}" = "ar" ]; then
    model_path="asafaya/bert-base-arabic"
  elif [ "${BERT_MODEL}" = "bg-cs-pl-ru" ]; then
    model_path="DeepPavlov/bert-base-bg-cs-pl-ru-cased"
  elif [ "${BERT_MODEL}" = "en" ]; then
    model_path="bert-base-cased"
  elif [ "${BERT_MODEL}" = "fi" ]; then
    model_path="TurkuNLP/bert-base-finnish-cased-v1"
  elif [ "${BERT_MODEL}" = "fr" ]; then
    model_path="camembert/camembert-base"
  elif [ "${BERT_MODEL}" = "it" ]; then
    model_path="dbmdz/bert-base-italian-cased" 
  elif [ "${BERT_MODEL}" = "mbert" ]; then
    model_path="bert-base-multilingual-cased" 
  elif [ "${BERT_MODEL}" = "nl" ]; then	  
    model_path="wietsedv/bert-base-dutch-cased"
  elif [ "${BERT_MODEL}" = "pl" ]; then	  
    model_path="dkleczek/bert-base-polish-uncased-v1"
  elif [ "${BERT_MODEL}" = "ru" ]; then
    model_path="DeepPavlov/rubert-base-cased"
  elif [ "${BERT_MODEL}" = "sv" ]; then
    model_path="KB/bert-base-swedish-cased"
  fi

  for filepath in ${TB_DIR}/*/${tbid}-ud-train.conllu; do
    dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
    tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

    # ud v2.x
    export TRAIN_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-train.conllu
    export DEV_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-dev.conllu
    export TEST_DATA_PATH=${TB_DIR}/${tb_name}/${tbid}-ud-test.conllu

    # AutoTokenizer model name
    export MODEL_NAME=${model_path}
  
    allennlp train configs/ud_${TASK}_bert_${MODEL_TYPE}_${FEATS}.jsonnet -s logs/${tbid}-${TASK}-${MODEL_TYPE}-${BERT_MODEL}-BERT-${FEATS}-${TIMESTAMP} --include-package tagging
  done
done

