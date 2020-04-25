#!/usr/bin/env bash

# usage: ./scripts/fix_all_final.sh tmp/final_tmp/ tmp/final_fixed/

# directory where you have copied the test files to
TMP_DIR=$1

# where to write fixed output
FIXED_DIR=$2

echo "searching ${TMP_DIR}"

for file in $(ls $TMP_DIR); do 
    echo "found $file"
    
    LCODE=$(echo ${file} | awk -F "_" '{print $2}')
    echo "using $LCODE"

    # adjust path to tools if necessary
    perl ${HOME}/tools/conllu-quick-fix.pl < $TMP_DIR/$file > $FIXED_DIR/$LCODE.conllu

done 

