# Standard library
import time

__all__ = ['Timer']

class Timer(object):

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args):
        self.time = self.elapsed()

    def reset(self):
        self.start = time.clock()

    def elapsed(self):
        return time.clock() - self.start
