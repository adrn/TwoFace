***************
Getting started
***************

Initializing and loading the database
=====================================

To get going on a new machine, you'll need to first download APOGEE data
(currently DR13) to a path, e.g., ``$APOGEE_PATH`` below. We can then initialize
the database by loading these data files:

.. code-block:: bash

    % export APOGEE_PATH="~/Data/APOGEE_DR13"
    % python initdb.py \
    > --allstar="$APOGEE_PATH/allStar-l30e.2.fits" \
    > --allvisit="$APOGEE_PATH/allVisit-l30e.2.fits" \
    > --redclump="$APOGEE_PATH/apogee-rc-DR13.fits" \
    > -vv

