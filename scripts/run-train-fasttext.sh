#!/bin/bash

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

  
#SBATCH -p compute       # which partition to run on
#SBATCH -J fasttext   # name for the job
#SBATCH --mem=161099
#SBATCH --exclusive      # TODO: our compiled version seems to only use 12 cores
#SBATCH --constraint=avx

# example: sbatch --export=ALL,FT_LANG=Russian,FT_LCODE=ru run-train-fasttext.sh

CONLL17_DIR=${HOME}/data/2-conll17
TMP_DIR=${CONLL17_DIR}/tmp
test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/enhanced-ud/Enhanced-UD-Parsing

SCRIPT_DIR=${PRJ_DIR}/scripts
TEXT_DIR=${CONLL17_DIR}/text

test -z $FT_LCODE && echo "Missing LCODE"
test -z $FT_LCODE && exit 1

test -z $FT_LANG && echo "Missing LANG"
test -z $FT_LANG && exit 1

echo "== $FT_LCODE $FT_LANG =="
date
hostname

cd ${TEXT_DIR}
hostname > model_$FT_LCODE.start
fasttext skipgram -minCount 5 -epoch 10 -neg 10 -input $FT_LANG.txt -output model_$FT_LCODE
touch model_$FT_LCODE.end

echo "Finished"
date
