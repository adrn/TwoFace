# Standard library
import time

__all__ = ['Timer']

class Timer(object):

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.time = self.end - self.start
