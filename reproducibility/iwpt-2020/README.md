Work in progress: We aim to have this documentation ready for the start of the conference.

## Basic Parser

The file `udpipe-future-models-used-in-dev.tsv` shows what models we used during development.
A `+` in the treebanks column means that a multi-treebank model was trained.
The prefix `task.` means that a treebank as distributed by the shared task was used.
The prefix `ud25.` means that a treebank from the UD v2.5 release was used.


## Semantic Parser

The file `allennlp-models-used-in-dev.txt` shows what models we used during development.
The type of model used can be inferred from the name of the model. For instance, `allennlp-090` means the model was produced with AllenNLP version `0.9.0` and `allennlp-dev` means the model was produced using the `development` branch.
The `treebank id` is given in the metadata string, e.g. `ar_padt`.
`enhanced` means the parser was trained to produce enhanced dependency graphs, i.e. the `deps` column in a CoNLL-U file.
`dm` means the parser uses a model inspired by Dozat and Manning (2018), using a BiLSTM encoder where the hidden representations are passed to four separate MLPs to obtain representations: `arc-head`, `arc-dep`, `rel-head` and  `rel-dep`. We then compute [Bilinear matrix attention](https://github.com/allenai/allennlp/blob/master/allennlp/modules/matrix_attention/bilinear_matrix_attention.py) between the respective arc and deprel representations and use sigmoid cross-entropy loss for arc-prediction and softmax cross-entropy loss for label prediction.
The type of `BERT` model is then given, e.g. `ar-BERT` means we used a monolingual Arabic BERT model.
Finally, the type of features (embeddings) used are given:
- `l`: lemma
- `u`: upos
- `x`: xpos
- `f`: morph feats
- `b`: basic tree (head direction and distance features as well as embedding the dependency relation from the basic tree).

The date is added to the metadata string to provide a unique identifier per each model and to help us keep track of model experiments.

To train your own enhanced dependency parsing models using features from a BERT model, see `scripts/train_wrapper_bert_stable.sh` and `scripts/train_wrapper_bert.sh` for training models using the `0.9.0` and `dev` branches. Note that results using BERT with the `dev` branch are lower for many languages than `0.9.0` at the time of the shared task deadline. This may not be the case with the current `dev` branch and we intend to carry out subsequent experiments to find out why this is the case.

TODO:
* create a similar file for test
* upload models somewhere (we have them on google drive already, maybe this location is ok? only needs to be made public then) and provide link

Note that there are two 20200419-230600 and -232103 models in the model folder but
we only used one each.
