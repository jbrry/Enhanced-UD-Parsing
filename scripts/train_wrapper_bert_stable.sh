#!/bin/bash

# wraps scripts/train.sh so multiple tbids can be run and if a tbid fails the script will continue.
# run: sbatch --gres=gpu:rtx2080ti:1 -J kg_graph scripts/train_wrapper.sh enhanced kg 1

test -z $1 && echo "Missing task type <basic> or <enhanced>"
test -z $1 && exit 1
TASK=$1

test -z $2 && echo "Missing model type <dm> or <kg>"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing job group <A:F>"
test -z $3 && exit 1
JOB=$3

test -z $4 && echo "Missing language code for BERT; currently supports: (ar bg_cs_pl_ru en fi it mbert nl pl ru sv)"
test -z $4 && exit 1
BERT_TYPE=$4

test -z $5 && echo "Missing seed number"
test -z $5 && exit 1
SEED=$5

# job groups based on training data size distributed in a round-robin fashion.
# run python utils/get_training_information.py to get training metadata.

# L-BERT
JOB_A="bg_btb cs_fictree en_ewt"
JOB_B="pl_pdb fr_sequoia nl_lassysmall nl_alpino"

# M-BERT
JOB_C="lv_lvtb lt_alksnis pl_lfg"
JOB_D="ta_ttb et_ewt uk_iu sk_snk"

# High memory
JOB_E="cs_pdt fi_tdt it_isdt"
JOB_F="ar_padt ru_syntagrus sv_talbanken cs_cac"
# fi_tdt


if [ ${JOB} == "A" ]; then
  JOB_GROUP=$JOB_A
elif [ ${JOB} == "B" ]; then
  JOB_GROUP=$JOB_B
elif [ ${JOB} == "C" ]; then
  JOB_GROUP=$JOB_C
elif [ ${JOB} == "D" ]; then
  JOB_GROUP=$JOB_D
elif [ ${JOB} == "E" ]; then
  JOB_GROUP=$JOB_E
elif [ ${JOB} == "F" ]; then
  JOB_GROUP=$JOB_F
fi 

for TBID in ${JOB_GROUP[@]}; do
    echo "training on ${TBID}"

    if [ "${BERT_TYPE}" == 'mbert' ]; then
      echo "using multilingual bert"
      BERT_LANG=mbert

    elif [ "${BERT_TYPE}" == 'lbert' ]; then
      echo "using language-specific bert"
      BERT_LANG=$(echo ${TBID} | awk -F "_" '{print $1}')
 
      if [ "$BERT_LANG" = "cs" ] || [ "$BERT_LANG" = "bg" ]; then
        BERT_LANG=bg_cs_pl_ru
      fi
    fi

    bash scripts/train_bert_stable.sh ${TASK} ${MODEL_TYPE} ${TBID} ${BERT_LANG} ${SEED}
done

