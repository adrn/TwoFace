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

For testing, it is sometimes useful to load smaller files. Sub-samples of the
full APOGEE data are provided with a fresh clone of this repo and can be
linked to (from within the ``twoface/scripts`` directory) with:

.. code-block:: bash

    % export APOGEE_PATH="../twoface/db/tests/"
    % python initdb.py \
    > --allstar="$APOGEE_PATH/test-allStar.fits" \
    > --allvisit="$APOGEE_PATH/test-allVisit.fits" \
    > --redclump="$APOGEE_PATH/test-rc.fits" \
    > -vv

Now, need to load Jason Cao's...

.. code-block:: bash

    % python load_cao.py \
    > --cao="$APOGEE_PATH/cao....fits" \
    > -vv

or, for testing:

.. code-block:: bash

    % python load_cao.py \
    > --cao="$APOGEE_PATH/test-cao.fits" \
    > -vv

Generating posterior samples for the ``RedClump`` table
=======================================================

