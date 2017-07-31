#!/bin/bash -x

set -ev

if [[ $BUILD_PAPER == true ]]; then
   source ci/setup-paper.sh
else
   git clone git://github.com/astropy/ci-helpers.git
   source ci-helpers/travis/setup_conda.sh
   if [[ $SETUP_CMD != *"egg_info"* ]]; then python setup.py install; fi
fi
