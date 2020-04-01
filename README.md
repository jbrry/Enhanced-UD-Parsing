# Enhanced UD Parsing

Repository for enhanced UD parsing for the 2020 [IWPT/UD Shared Task](https://universaldependencies.org/iwpt20/).

## Table of Contents

- [Installation](#installation)
- [Obtain data](#obtain-data)
- [Train models](#train-models)

## Installation

If using a [Conda](https://conda.io/) environment, create a Conda environment with Python 3.7:

```bash
conda create -n enhanced_parsing python=3.7
```

Activate the Conda environment:

```bash
conda activate enhanced_parsing
```

This project uses AllenNLP [installed from the master branch](https://github.com/allenai/allennlp#installing-from-source) and the most recent version of the [conllu](https://github.com/EmilStenstrom/conllu) library. If you install AllenNLP from master you should have the required version.

## Obtain data
You will need to obtain the official shared task training and development data:

```bash
cd data
wget http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-train-dev.tgz
tar -xvzf iwpt2020-train-dev.tgz
```

## Train models
Currently, this project is still in development but you can run the most-recent version of the graph-parser(s) as below. The filepaths in the configuration file can be adjusted for a particular language or for quicker debugging you can use the en_ewt files as below:

```bash
# optional
export TRAIN_DATA_PATH=data/UD_English-EWT/en_ewt-ud-train_no_ellipsis.conllu
export DEV_DATA_PATH=data/UD_English-EWT/en_ewt-ud-dev_no_ellipsis.conllu
```

To train a graph-based parser with Kiperwasser and Goldberg (2016) scoring function, run:

```bash
allennlp train -f configs/ud_enhanced_kg.jsonnet -s logs/kg_graph --include-package tagging
```

To train a graph-based parser with Dozat and Manning (2016) scoring function, run:

```bash
allennlp train -f configs/ud_enhanced_dm.jsonnet -s logs/dm_graph --include-package tagging
```

Alternatively, you can use `train.sh` with the following arguments, where `${task_type}` is either `basic` or `enhanced`, `${model_type}` is either `dm` or `kg` depending on the type of parser being used, and `${tbid}` is the treebank id, e.g. `en_ewt`.

```bash
./scripts/train.sh ${task_type} ${model_type} ${tbid}
```
