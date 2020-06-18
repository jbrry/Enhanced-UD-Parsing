#!/bin/bash

# downloads regular BERT/Multilingual BERT and changes to PyTorch format

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: James Barry

TRANSFORMER_DIR=${HOME}/transformer_dir/pytorch_models
cd $TRANSFORMER_DIR

LANGS="en mbert"

for lang in $LANGS;do
  echo "== $lang =="
  mkdir -p $lang
  cd $lang
  
  # BERT
  if [ "${lang}" = "en" ]; then
    echo "downloading $lang BERT model"
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    unzip cased_L-12_H-768_A-12.zip
    rm cased_L-12_H-768_A-12.zip

    # use transformers library to convert to pytorch
    python -m transformers.convert_bert_original_tf_checkpoint_to_pytorch \
	    --tf_checkpoint_path cased_L-12_H-768_A-12/bert_model.ckpt \
	    --bert_config_file cased_L-12_H-768_A-12/bert_config.json \
	    --pytorch_dump_path pytorch_model.bin

    cp cased_L-12_H-768_A-12/bert_config.json .
    cp cased_L-12_H-768_A-12/vocab.txt .
    tar -czf weights.tar.gz bert_config.json pytorch_model.bin

  # mBERT
  elif [ "${lang}" = "mbert" ]; then
    echo "downloading $lang BERT model"
    wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
    unzip multi_cased_L-12_H-768_A-12.zip
    rm multi_cased_L-12_H-768_A-12.zip

    # use transformers library to convert to pytorch
    python -m transformers.convert_bert_original_tf_checkpoint_to_pytorch \
      --tf_checkpoint_path multi_cased_L-12_H-768_A-12/bert_model.ckpt \
      --bert_config_file multi_cased_L-12_H-768_A-12/bert_config.json \
      --pytorch_dump_path pytorch_model.bin

    cp multi_cased_L-12_H-768_A-12/bert_config.json .
    cp multi_cased_L-12_H-768_A-12/vocab.txt .
    tar -czf weights.tar.gz bert_config.json pytorch_model.bin

  fi

  cd ${TRANSFORMER_DIR}
done
