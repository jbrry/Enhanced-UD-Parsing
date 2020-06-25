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

### If using `venv`, to create environments with stable and bleeding edge:

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
You can use `train.sh` with the following arguments, where `${task_type}` is either `basic` or `enhanced`, `${model_type}` is either `dm` or `kg` depending on the type of parser being used, `${tbid}` is the treebank id, e.g. `en_ewt`, `${random_seed}` is the random seed e.g. `12345` and `${package}` is the package version, either `tagging_stable` or `tagging`.

```bash
./scripts/train.sh ${task_type} ${model_type} ${tbid} ${random_seed} ${package}
```

## Basic Parser

* https://github.com/jowagner/UDPipe-Future/tree/multitreebank (use the `tbemb` branch, e.g. with `git checkout tbemb`)
* https://github.com/jowagner/ud-combination


## Citing
If you wish to cite this paper or if you use the software in your research please use the reference below:

```latex
@InProceedings{barry-wagner-foster:2020:iwpt,
  author    = {Barry, James  and  Wagner, Joachim  and  Foster, Jennifer},
  title     = {The {ADAPT} Enhanced Dependency Parser at the {IWPT} 2020 Shared Task},
  booktitle      = {Proceedings of the 16th International Conference on Parsing Technologies and the IWPT 2020 Shared Task on Parsing into Enhanced Universal Dependencies},
  month          = jul,
  year           = {2020},
  address        = {Online},
  publisher      = {Association for Computational Linguistics},
  pages     = {227--235},
  abstract  = {We describe the ADAPT system for the 2020 IWPT Shared Task on parsing enhanced Universal Dependencies in 17 languages. We implement a pipeline approach using UDPipe and UDPipe-future to provide initial levels of annotation. The enhanced dependency graph is either produced by a graph-based semantic dependency parser or is built from the basic tree using a small set of heuristics. Our results show that, for the majority of languages, a semantic dependency parser can be successfully applied to the task of parsing enhanced dependencies. Unfortunately, we did not ensure a connected graph as part of our pipeline approach and our competition submission relied on a last-minute fix to pass the validation script which harmed our official evaluation scores significantly. Our submission ranked eighth in the official evaluation with a macro-averaged coarse ELAS F1 of 67.23 and a treebank average of 67.49. We later implemented our own graph-connecting fix which resulted in a score of 79.53 (language average) or 79.76 (treebank average), which would have placed fourth in the competition evaluation.},
  url       = {https://www.aclweb.org/anthology/2020.iwpt-1.24}
}
```
