*************************
Running the tests locally
*************************

Dependencies
============

- `postgres <>`_
- ...


Configuration
=============

In order to run the full suite of tests, you may have to provide database
connection credentials. By default, ``TwoFace`` assumes the user is ``postgres``
with no password, and in the process of running the tests will create a database
called ``twoface_test`` in the postgresql server running on port 5432 at
the host ``localhost``. All of these things are configurable. To change these
defaults, you must create or edit the file ``~/.twoface_config``. Here is an
example config file with the above defaults:

.. code-block:: cfg

    [testing]
    user = postgres
    database = twoface_test
    host = localhost
    port = 5432
    password = None
