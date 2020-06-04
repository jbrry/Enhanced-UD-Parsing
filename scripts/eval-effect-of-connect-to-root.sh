#!/bin/bash

if [ -n $UD_TOOLS_DIR ] ; then
    echo "Detected UD tools dir"
else
    echo "Error: UD_TOOLS_DIR not set"
    exit 1
fi

if [ -e scripts/wrapper-collapse-empty-nodes.sh ] ; then
    echo "Detected usable repo dir"
else
    cd $HOME/enhanced-ud/Enhanced-UD-Parsing
fi
    
# collapse

for I in $HOME/enhanced-ud/effect-of-connect-to-root/[bdn]*/pred*u ; do
    scripts/wrapper-collapse-empty-nodes.sh $I $I.collapsed
done

# eval

for TBID in cs_cac en_ewt it_isdt lt_alksnis ; do
    for I in $HOME/enhanced-ud/effect-of-connect-to-root/[bdn]*/pred*${TBID}*ed ; do

        scripts/iwpt20_xud_eval.py --output $I.eval.txt --verbose \
            $HOME/enhanced-ud/Enhanced-UD-Parsing/data/predictions/collapsed/${TBID}-ud-dev.conllu \
            $I

    done
done

