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

### Supporting prediction with both stable and bleeding edge

Here we are testing with Python 3.6.

#### Virtual environment for stable 0.9.0

Check that 0.9.0 is the most recent stable. 1.0.0 could be there any moment now.

```
mkdir venv
cd venv
virtualenv -p /usr/bin/python3.6 allennlp-090
```
If needed, edit `allennlp-090/bin/activate` to configure your CUDA environment, e.g.
```
# Manually added configuration
LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:"$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
```

Activate and install packages:
```
source allennlp-090/bin/activate
pip install torch torchvision
pip install cython
pip install allennlp
pip install -U conllu
exit
```

#### Virtual environment for bleeding edge 

```
mkdir venv
cd venv
virtualenv -p /usr/bin/python3.6 allennlp-dev
```
If needed, edit `allennlp-dev/bin/activate` to configure your CUDA environment, e.g.
```
# Manually added configuration
LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:"$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
```

Activate and install packages:
```
source allennlp-dev/bin/activate
pip install torch torchvision
pip install cython
pip install allennlp==1.0.0.dev20200418
pip install -U conllu
exit
```

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
