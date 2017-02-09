#!/bin/bash -x

set -ev

if [[ $BUILD_PAPER != true ]]; then
   export TWOFACE_CACHE_PATH=~/twoface_test/

   if [[ $SETUP_CMD != *"egg_info"* ]]; then
       python scripts/initdb.py --test -v
   fi
fi
