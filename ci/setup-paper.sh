#!/bin/bash -x

echo "Building the paper..."
export BUILDING_PAPER=true
source "$( dirname "${BASH_SOURCE[0]}" )"/setup-texlive.sh
