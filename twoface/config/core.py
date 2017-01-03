# Standard library
import os
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

# Third-party
from astropy.utils.data import get_pkg_data_filename

conf = ConfigParser()
conf.read(get_pkg_data_filename("defaults.cfg"))
conf.read(os.path.abspath(os.path.expanduser("~/.twoface_config")))
