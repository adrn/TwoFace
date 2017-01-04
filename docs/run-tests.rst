*************************
Running the tests locally
*************************

Start by cloning the repository and change into the cloned repo:

.. code-block:: bash

    git clone git@github.com:adrn/TwoFace.git
    cd TwoFace

Dependencies
============

- `postgres <>`_
- ...

The Python package dependencies are included in a `conda
<https://www.continuum.io/downloads>`_ `environment file
<https://github.com/adrn/TwoFace/blob/master/environment.yml>`_ -- I recommend
creating a new environment to install ``TwoFace`` and these dependencies. You
can automatically create an environment called ``twoface`` by running:

.. code-block:: bash

    conda env create

in a cloned version of the repository

Configuration
=============

In order to run the full suite of tests, you may have to provide database
connection credentials. By default, ``TwoFace`` assumes the user is ``postgres``
with no password, and in the process of running the tests will create a database
called ``twoface_test`` in the postgresql server running on port 5432 at
the host ``localhost``. All of these things are configurable. You can also opt
to skip the database tests and control whether the test database is deleted
after the tests complete. To change these defaults, you must create or edit the
file ``~/.twoface_config``. Here is an example config file of the defaults:

.. code-block:: cfg

    [testing]
    user = postgres
    database = twoface_test
    host = localhost
    port = 5432
    password =
    skip_db_tests = False
    delete_test_db = True

Running the tests
=================



