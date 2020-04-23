#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

SOURCE=copy_parse.py

for EXTRA in     _rel             _cc             _cc_rel \
    _mark        _mark_rel        _mark_cc        _marc_cc_rel \
    _encase      _encase_rel      _encase_cc      _encase_cc_rel \
    _encase_mark _encase_mark_rel _encase_mark_cc _encase_mark_cc_rel \
    _arcase      _arcase_rel      _arcase_cc      _arcase_cc_rel \
    _arcase_mark _arcase_mark_rel _arcase_mark_cc _arcase_mark_cc_rel ; do

                TARGET=copy_parse${EXTRA}.py
                if [ -e ${TARGET} ]; then
                    echo "${TARGET} already there, nothing to do"
                else
                    ln -s ${SOURCE} ${TARGET}
                fi

done

