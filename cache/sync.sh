#!/bin/bash

rsync -zvr --max-size=3000m --exclude "*~" \
perseus:/tigress/adrianp/projects/twoface/cache ~/projects/twoface
