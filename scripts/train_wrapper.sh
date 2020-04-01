#!/bin/bash

# wraps scripts/train.sh so multiple tbids can be run and if a tbid fails the script will continue.
# run: sbatch --gres=gpu:rtx2080ti:1 -J kg_graph scripts/train_wrapper.sh enhanced kg

test -z $1 && echo "Missing task type <basic> or <enhanced>"
test -z $1 && exit 1
TASK=$1

test -z $2 && echo "Missing model type <dm> or <kg>"
test -z $2 && exit 1
MODEL_TYPE=$2

for TBID in `cat scripts/tbids.txt`; do
    msg "training on ${TBID}"
    
    ./scripts/train.sh ${TASK} ${MODEL_TYPE} ${TBID}
done

