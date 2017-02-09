#!/bin/bash -x

if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'paper/'
then
  echo "Building the paper..."
  source "$( dirname "${BASH_SOURCE[0]}" )"/setup-texlive.sh
  return
fi
export BUILD_PAPER='true'
return
