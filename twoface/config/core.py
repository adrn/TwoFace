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

user_config_path = os.path.abspath(os.path.expanduser("~/.config/twoface/twoface.cfg"))
if os.path.exists(user_config_path):
    conf.read(user_config_path)
