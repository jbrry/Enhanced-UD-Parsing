#!/usr/bin/env bash

# usage: ./scripts/fix_all_final.sh tmp/final_tmp/ tmp/final_fixed/


# directory where you have copied the test files to
FINAL_DIR=$1

# temp dir
TMP_DIR=$2

# where to write fixed output
FIXED_DIR=$3

echo "searching ${FINAL_DIR}"

for file in $(ls $FINAL_DIR); do 
    echo "found $file"

    echo "copying $file to $TMP_DIR"
    cp $FINAL_DIR/$file $TMP_DIR/

    rm $TMP_DIR/tmp.conllu
    
    cut -f -10 $TMP_DIR/$file > $TMP_DIR/tmp.conllu
    rm $TMP_DIR/$file
    
    cp $TMP_DIR/tmp.conllu $TMP_DIR/$file 
    
    LCODE=$(echo ${file} | awk -F "_" '{print $2}')
    echo "using $LCODE"

    # adjust path to tools if necessary
    perl ${HOME}/tools/conllu-quick-fix.pl --connect-to-root < $TMP_DIR/$file > $FIXED_DIR/$LCODE.conllu

done 

