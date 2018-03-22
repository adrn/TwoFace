__all__ = ['GoogleSheets', 'FitsTable']

from astropy.table import Table
from astropy.io import fits


class BaseFileLoader():
    def __init__(self):
        pass

    def _load(self):
        return

    def load(self):
        if not hasattr(self, '_data'):
            self._data = self._load()
        return self._data


class GoogleSheets(BaseFileLoader):
    def __init__(self, key, gid, **kwargs):
        self._url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
        self._kwargs = kwargs

    def _load(self):
        return Table.read(self._url, format='ascii.csv', **self._kwargs)


class FitsTable(BaseFileLoader):
    def __init__(self, path):
        self._path = path

    def _load(self):
        hdulist = fits.open(self._path)
        data = hdulist[1].data
        hdulist.close()
        return data
