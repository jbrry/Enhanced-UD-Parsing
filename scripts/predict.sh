#!/bin/bash

allennlp predict /home/jbarry/allennlp-tutorial-practice/logs/e_parser_gpu/model.tar.gz data/UD_English-EWT/en_ewt-ud-dev_no_ellipsis.conllu --output-file x.conllu --predictor enhanced-predictor --include-package tagging --use-dataset-reader
