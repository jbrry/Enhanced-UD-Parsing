#!/bin/bash

# downloads BERT models for languages with pre-trained models on https://huggingface.co/models
# assumes regular BERT/Multilingual BERT are already downloaded and converted to pytorch format
# TODO French Roberta?

TRANSFORMER_DIR=${HOME}/transformer_dir/pytorch_models
cd $TRANSFORMER_DIR

LANGS="ar bg-cs-pl-ru fi it nl pl sv"

for lang in $LANGS;do
  echo "== $lang =="
  mkdir -p $lang
  cd $lang

  # get user and model names on huggingface
  if [ "${lang}" = "ar" ]; then
    model_path="asafaya/bert-base-arabic"
  elif [ "${lang}" = "bg-cs-pl-ru" ]; then
    model_path="DeepPavlov/bert-base-bg-cs-pl-ru-cased"
  elif [ "${lang}" = "fi" ]; then
    model_path="TurkuNLP/bert-base-finnish-cased-v1"
  elif [ "${lang}" = "it" ]; then
    model_path="dbmdz/bert-base-italian-cased" 
  elif [ "${lang}" = "nl" ]; then	  
    model_path="wietsedv/bert-base-dutch-cased"
  elif [ "${lang}" = "pl" ]; then	  
    model_path="dkleczek/bert-base-polish-uncased-v1"
  elif [ "${lang}" = "sv" ]; then
    model_path="KB/bert-base-swedish-cased"
  fi

  echo "Using model path $model_path"

  user=`echo $model_path | awk -F/ '{print $1}'`
  model=`echo $model_path | awk -F/ '{print $2}'`

  echo "Downloading models from huggingface with the following credentials:"
  echo "user - $user"
  echo "model - $model"

  # download the files from huggingface
  wget https://cdn.huggingface.co/$user/$model/config.json
  wget https://cdn.huggingface.co/$user/$model/vocab.txt
  wget https://cdn.huggingface.co/$user/$model/pytorch_model.bin

  cp config.json bert_config.json
  tar -czf weights.tar.gz bert_config.json pytorch_model.bin

  echo "Finished \n"
  cd ${TRANSFORMER_DIR}
done

