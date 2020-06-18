#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner


SOURCE=allennlp_090_dm_mbert_u.py

for VERSION in 090 dev ; do
    for PTYPE in dm kg ; do
        for BERT in mbert lbert pbert ; do
            for FEATS in u luf lufb lux luxf luxfb ; do
                TARGET=allennlp_${VERSION}_${PTYPE}_${BERT}_${FEATS}.py
                if [ -e ${TARGET} ]; then
                    echo "${TARGET} already there, nothing to do"
                else
                    ln -s ${SOURCE} ${TARGET}
                fi
            done
        done
    done
done

