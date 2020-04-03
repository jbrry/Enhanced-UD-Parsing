#!/bin/bash

# wraps scripts/train.sh so multiple tbids can be run and if a tbid fails the script will continue.
# run: sbatch --gres=gpu:rtx2080ti:1 -J kg_graph scripts/train_wrapper.sh enhanced kg 1

test -z $1 && echo "Missing task type <basic> or <enhanced>"
test -z $1 && exit 1
TASK=$1

test -z $2 && echo "Missing model type <dm> or <kg>"
test -z $2 && exit 1
MODEL_TYPE=$2

test -z $3 && echo "Missing job group number <1:5>"
test -z $3 && exit 1
JOB=$3

# job groups based on training data size distributed in a round-robin fashion.
# run python utils/get_training_information.py to get training metadata.
JOB_1="cs_pdt it_isdt lv_lvtb uk_iu ta_ttb"
JOB_2="ru_syntagrus en_ewt bg_btb sv_talbanken"
JOB_3="cs_cac nl_alpino sk_snk lt_alksnis"
JOB_4="pl_pdb fi_tdt ar_padt fr_sequoia"
JOB_5="pl_lfg cs_fictree nl_lassysmall et_ewt"

if [ $JOB == 1 ]; then
  JOB_GROUP=$JOB_1
elif [ $JOB == 2 ]; then
  JOB_GROUP=$JOB_2
elif [ $JOB == 3 ]; then
  JOB_GROUP=$JOB_3
elif [ $JOB == 4 ]; then
  JOB_GROUP=$JOB_4
elif [ $JOB == 5 ]; then
  JOB_GROUP=$JOB_5
fi 
       
for TBID in "${JOB_GROUP[*]}"; do
    echo "training on ${TBID}"

    bash scripts/train.sh ${TASK} ${MODEL_TYPE} ${TBID}
done

