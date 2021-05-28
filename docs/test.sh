#!/usr/bin/env bash

# Fetch docs code
git clone https://github.com/Epistimio/orion.git
cd orion/
git checkout master

pip install -e .
# Replace remote include by local include to test local doc version 
sed -i 's/.. plugin-include:: https:\/\/raw.githubusercontent.com\/Epistimio\/orion.algo.skopt\/master/.. include:: ..\/..\/..\/../g' docs/src/user/algorithms.rst
# Test
tox -e docs
