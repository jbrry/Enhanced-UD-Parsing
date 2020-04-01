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
This project uses AllenNLP [installed from the master branch](https://github.com/allenai/allennlp#installing-from-source) and the most recent version of the [conllu](https://github.com/EmilStenstrom/conllu) library. If you are not using the most recent version of the conllu library, there will be some errors parsing the data. Install the requirements:

## Obtain data
You will need to obtain the official shared task training and development data:

```
cd data
wget http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-train-dev.tgz
tar -xvzf iwpt2020-train-dev.tgz
```

## Train models
Currently, this project is still in development but you can run the most-recent version of the graph-parser(s) as below. The filepaths in the configuration file can be adjusted for a particular language or for quicker debugging you can use the en_ewt files as below:

```
# optional
export TRAIN_DATA_PATH=data/UD_English-EWT/en_ewt-ud-train_no_ellipsis.conllu
export DEV_DATA_PATH=data/UD_English-EWT/en_ewt-ud-dev_no_ellipsis.conllu
```

To train a graph-based parser with Kiperwasser and Goldberg (2016) scoring function, run:

```
allennlp train -f configs/ud_enhanced_kg.jsonnet -s logs/kg_graph --include-package tagging
```

To train a graph-based parser with Dozat and Manning (2016) scoring function, run:
```
allennlp train -f configs/ud_enhanced_dm.jsonnet -s logs/dm_graph --include-package tagging
```

Alternatively, you can edit `train.sh` to point to `configs/enhanced_parser_${model_type}.jsonnet` and the new shared task data.
