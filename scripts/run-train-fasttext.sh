#!/bin/bash
  
#SBATCH -p compute       # which partition to run on
#SBATCH -J fasttext   # name for the job
#SBATCH --mem=128000
#SBATCH --exclusive      # TODO: our compiled version seems to only use 12 cores
#SBATCH --constraint=avx

CONLL17_DIR=${HOME}/data/2-conll17
TMP_DIR=${CONLL17_DIR}/tmp
test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/enhanced-ud/Enhanced-UD-Parsing

SCRIPT_DIR=${PRJ_DIR}/scripts
TEXT_DIR=${CONLL17_DIR}/text

test -z $LCODE && echo "Missing LCODE"
test -z $LCODE && exit 1

test -z $LANG && echo "Missing LANG"
test -z $LANG && exit 1

echo "== $LCODE $LANG =="
date

cd ${TEXT_DIR}
hostname > model_$LCODE.start
fasttext skipgram -minCount 5 -epoch 10 -neg 10 -input $LANG.txt -output model_$LCODE
touch model_$LCODE.end

echo "Finished"
date
