.. _install-config:

******************************
Installation and configuration
******************************

Start by cloning the repository and change into the cloned repo:

.. code-block:: bash

    git clone git@github.com:adrn/TwoFace.git
    cd TwoFace

Dependencies
============

First off, we require Python >= 3.5

The Python package dependencies are included in a `conda
<https://www.continuum.io/downloads>`_ `environment file
<https://github.com/adrn/TwoFace/blob/master/environment.yml>`_ -- I recommend
creating a new environment to install ``TwoFace`` and these dependencies. You
can automatically create an environment called ``twoface`` by running:

.. code-block:: bash

    conda env create -f environment.yml

in the cloned version of the repository.

Configuration
=============

Most of the processing in this package builds and operates on an SQLite
database. This database and any generated files are stored in a cache location,
by default ``~/.twoface/``. This location can be configured by setting the
environment variable ``$TWOFACE_CACHE_PATH``. For example, with a Bash shell,
to explicitly set the cache path to the default path:

.. code-block:: bash

    export TWOFACE_CACHE_PATH=~/.twoface/

This path is also the default location to store cached prior samples. These
files are stored as HDF5 files.
