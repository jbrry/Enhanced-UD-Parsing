#!/bin/bash

# make connect_graph work with stdin and stdout

SCRIPT_DIR=$(dirname $0)

TEMP_DIR=$(mktemp -d eud-fragment-fix.XXXXXX)

python3 $SCRIPT_DIR/connect_graph.py -i /dev/stdin -o $TEMP_DIR

cat $TEMP_DIR/*

rm -rf $TEMP_DIR

