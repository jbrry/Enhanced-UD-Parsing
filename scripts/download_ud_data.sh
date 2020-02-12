#!/usr/bin/env bash

# source: https://github.com/Hyperparticle/udify/blob/master/scripts/download_ud_data.sh

# Can download UD 2.3 or 2.4
UD_2_3="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/ud-treebanks-v2.3.tgz?sequence=1&isAllowed=y"
UD_2_4="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz?sequence=4&isAllowed=y"
UD_2_5="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz?sequence=1&isAllowed=y"

DATASET_DIR="data/ud-treebanks-v2.5"

echo "Downloading and unpacking UD data..."

curl ${UD_2_5} -o - | tar -xvzf - -C ./data

#echo "Generating multilingual dataset..."
#mkdir -p "data/ud"
#mkdir -p "data/ud/multilingual"
#python concat_treebanks.py data/ud/multilingual --dataset_dir ${DATASET_DIR}
