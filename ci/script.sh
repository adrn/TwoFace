#!/bin/bash -x

set -ev

echo $BUILDING_PAPER

if [[ $BUILD_PAPER == true ]]; then
   source ci/build-paper.sh
else
   $MAIN_CMD $SETUP_CMD
fi
