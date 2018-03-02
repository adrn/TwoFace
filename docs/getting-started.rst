***************
Getting started
***************

Initializing and loading the database
=====================================

To get going on a new machine, you'll need to first download APOGEE data
(currently DR13) to a path, e.g., ``$APOGEE_PATH`` below. Aside from the allStar
and allVisit files, you'll also need the value-added Red Clump catalog, and a
catalog of red giant masses from `Ness et al. (2016)
<http://iopscience.iop.org/article/10.3847/0004-637X/823/2/114/meta>`_. We can
then initialize the database by loading these data files:

.. code-block:: bash

    % export APOGEE_PATH="~/data/APOGEE_DR14"
    % python initdb.py \
    > --allstar="$APOGEE_PATH/allStar-l31c.2.fits" \
    > --allvisit="$APOGEE_PATH/allVisit-l31c.2.fits" \
    > --nessrg="$APOGEE_PATH/NessRG.fits" \
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
