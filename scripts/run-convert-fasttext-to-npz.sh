#!/bin/bash
  
#SBATCH -p compute       # which partition to run on
#SBATCH -J ft-2-npz   # name for the job
#SBATCH --mem=18000
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH -N 1-1

# example: sbatch --export=ALL,FT_LCODE=ru run-convert-fasttext-to-npz.sh

CONLL17_DIR=${HOME}/data/2-conll17
TMP_DIR=${CONLL17_DIR}/tmp
test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/enhanced-ud/Enhanced-UD-Parsing

SCRIPT_DIR=${PRJ_DIR}/scripts
TEXT_DIR=${CONLL17_DIR}/text

source ${PRJ_DIR}/../UDPipe-Future/venv-udpf/bin/activate

test -z $FT_LCODE && echo "Missing LCODE"
test -z $FT_LCODE && exit 1

echo "== $FT_LCODE =="
date

cd ${TEXT_DIR}
hostname > convert_$FT_LCODE.start

python3 ${PRJ_DIR}/../UDPipe-Future/embeddings/vec_to_npz/convert.py \
    --max_words 1000000 model_$FT_LCODE.vec fasttext-$FT_LCODE.npz

touch convert_$FT_LCODE.end

echo "Finished"
date
