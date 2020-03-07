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
This project uses AllenNLP and the most recent version of the [conllu](https://github.com/EmilStenstrom/conllu) library. If you are not using the most recent version of the conllu library, there will be some errors parsing the data. Install the requirements:

```bash
pip install -r ./requirements.txt
```

## Obtain data
You will need to obtain the official shared task training and development data:

```
cd data
wget http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-train-dev.tgz
tar -xvzf iwpt2020-train-dev.tgz
```

## Train models
Currently, this project is still in development but you can run the most recent version of the graph-parser as below. The filepaths in the configuration file can be adjusted for a particular language. Alternatively, you can edit `train.sh` to point to `configs/enhanced_parser.jsonnet` and the new shared task data.

```
allennlp train -f configs/enhanced_parser.jsonnet -s logs/enhanced_parser/ --include-package tagging
```


The `enhanced_parser` parser is a graph parser similar to that in [Dozat and Manning, 2018](https://www.aclweb.org/anthology/P18-2077/) which uses a sigmoid function to choose edges between all pairs of words above a certain threshold (and picks the most probable parent in cases where a word wasn't assigned a parent). This parser can compute labeled and unlabeled f1 and gets around 83% LF1 on `en_ewt-ud-dev_no_ellipsis.conllu`: https://tensorboard.dev/experiment/8UZQ5QHVRb2mTBGySyRmig/#scalars

`biaffine_parser_enhanced` is a multi-task parser, i.e. computes the basic tree as well as the enhanced graph and sums their respective losses but training is unstable: https://tensorboard.dev/experiment/wqLyYh4aTRKcKPJei9GtRQ/#scalars
It might need separate encoders e.g. shared and task-specific encoders or else some more debugging.
