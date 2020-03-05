#!/bin/bash

mkdir -p output

GOLD="data/UD_English-EWT/en_ewt-ud-dev_no_ellipsis.conllu"
PRED="output/en_ewt-ud-dev_no_ellipsis_pred.conllu"

MODEL_DIR=logs/enhanced_parser_10_epochs_with_full_decoding

#=== Predict ===
allennlp predict ${MODEL_DIR}/model.tar.gz ${GOLD} \
	--output-file ${PRED} \
       	--predictor enhanced-predictor \
	--include-package tagging \
	--use-dataset-reader

echo "Running UD Shared Task evaluation script..."
python scripts/iwpt20_xud_eval.py --verbose $GOLD $PRED

