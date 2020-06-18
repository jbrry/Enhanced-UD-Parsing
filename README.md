# Enhanced UD Parsing

Repository for enhanced UD parsing for the 2020 [IWPT/UD Shared Task](https://universaldependencies.org/iwpt20/).
Team ADAPT.

## Semantic Parser Installation 

This project contains the modules `tagging` and `tagging_stable` which use the master branch of AllenNLP and `AllenNLP 0.9.0` respectively.

If using a [Conda](https://conda.io/) environment, create a Conda environment with Python 3.7:

```bash
# stable
conda create -n enhanced_parsing_stable python=3.7
# dev
conda create -n enhanced_parsing_dev python=3.7
```
Activate the Conda environment and install the dependencies:

```bash
conda activate enhanced_parsing_stable
# or for dev
conda activate enhanced_parsing_dev
```

```bash
pip install -r requirements_stable.txt
# or for dev
pip install -r requirements_dev.txt
```

### Or if using `venv` with both stable and bleeding edge

Here we are testing with Python 3.6.

#### Virtual environment for stable 0.9.0

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

## Train semantic parser models
You can use `train.sh` with the following arguments, where `${task_type}` is either `basic` or `enhanced`, `${model_type}` is either `dm` or `kg` depending on the type of parser being used, `${tbid}` is the treebank id, e.g. `en_ewt` and `${random_seed}` is the random seed e.g. `12345`.

```bash
./scripts/train.sh ${task_type} ${model_type} ${tbid} ${random_seed}
```

## Basic Parser

* https://github.com/jowagner/UDPipe-Future/tree/multitreebank (use the `tbemb` branch, e.g. with `git checkout tbemb`)
* https://github.com/jowagner/ud-combination

