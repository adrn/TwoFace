*************************
Running the tests locally
*************************

Start by cloning the repository and change into the cloned repo:

.. code-block:: bash

    git clone git@github.com:adrn/TwoFace.git
    cd TwoFace

Dependencies
============

The Python package dependencies are included in a `conda
<https://www.continuum.io/downloads>`_ `environment file
<https://github.com/adrn/TwoFace/blob/master/environment.yml>`_ -- I recommend
creating a new environment to install ``TwoFace`` and these dependencies. You
can automatically create an environment called ``twoface`` by running:

.. code-block:: bash

    conda env create -f environment.yml

in the cloned version of the repository.

Running the tests
=================

The tests can then be run with:

.. code-block:: bash

    python setup.py test
