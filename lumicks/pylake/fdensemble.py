from lumicks.pylake.detail.alignment import align_fd_simple
import numpy as np


class FdEnsemble:
    def __init__(self, fd_curves):
        self.fd_curves = fd_curves
        self.fd_curves_processed = fd_curves

    def __getitem__(self, item):
        return self.fd_curves_processed[item]

    def __iter__(self):
        return self.fd_curves_processed.__iter__()

    def items(self):
        return self.fd_curves_processed.items()

    def values(self):
        return self.fd_curves_processed.values()

    def keys(self):
        return self.fd_curves_processed.keys()

    @property
    def raw(self):
        return self.fd_curves

    @property
    def f(self):
        return np.hstack(fd.f.data for fd in self.values())

    @property
    def d(self):
        return np.hstack(fd.d.data for fd in self.values())

