#!/bin/bash

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing input conllu file"
test -z $1 && exit 1
INFILE=$1

test -z $2 && echo "Missing output conllu file"
test -z $2 && exit 1
OUTFILE=$2

cd ${UD_TOOLS_DIR}
perl enhanced_collapse_empty_nodes.pl $INFILE > $OUTFILE

