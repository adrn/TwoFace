#!/bin/bash -x

# If building the paper, do that here
if [[ $BUILD_PAPER == paper ]]; then
  if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'paper/'
  then
    echo "Building the paper..."
    source "$( dirname "${BASH_SOURCE[0]}" )"/setup-texlive.sh
    return
  fi
  return
fi
