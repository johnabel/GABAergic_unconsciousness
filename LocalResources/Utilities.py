"""
A mixed file containing some utilities that I find useful in solving
circadian problems, updated to add anesthesiology utilities.

jha
"""

#import modules
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (splrep, splint, fitpack, splev,
                               UnivariateSpline, dfitpack,
                               InterpolatedUnivariateSpline)
from sklearn import decomposition
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
from time import time


def roots(data, times=None):
    """
    Takes a set of data and finds the roots of it. Uses a spline
    interpolation to get the root values.
    """

    if times is None:
        # time intervals set to one
        times = np.arange(len(data))

    # fits a spline centered on those indexes
    s = UnivariateSpline(times, data, s=0)

    return s.roots()


class laptimer:
    """
    Whenever you call it, it times laps.
    """

    def __init__(self):
        self.time = time()

    def __call__(self):
        ret = time() - self.time
        self.time = time()
        return ret

    def __str__(self):
        return "%.3E" % self()

    def __repr__(self):
        return "%.3E" % self()


class fnlist(list):
    def __call__(self, *args, **kwargs):
        return np.array([entry(*args, **kwargs) for entry in self])


def corrsort(mat):
    """ Function to sort correlation matrix so that correlated variables
    are closer to eachother.  """

    # Get the eigenvalues and eigenvectors, sort them for increasing
    # order
    w, v = np.linalg.eig(mat)
    v = v[:, w.argsort()]
    w.sort()

    e1 = v[:, -1]
    e2 = v[:, -2]

    angles = np.arctan(e2 / e1) + ~(e1 > 0) * np.pi
    order = angles.argsort()
    angles = angles[order]
    maxdiff = np.diff(angles).argmax() + 1

    # Expand at maximum angular difference
    order = np.concatenate((order[maxdiff:], order[:maxdiff]))

    # reorder matrix rows, columns
    mat = mat[order][:, order]

    return mat, order


def bode(G, f=np.arange(.01, 100, .01), desc=None, color=None):

    jw = 2 * np.pi * f * 1j
    y = np.polyval(G.num, jw) / np.polyval(G.den, jw)
    mag = 20.0 * np.log10(abs(y))
    phase = np.arctan2(y.imag, y.real) * 180.0 / np.pi % 360

    #plt.semilogx(jw.imag, mag)
    plt.semilogx(f, mag, label=desc, color=color)

    return mag, phase


class PeriodicSpline(UnivariateSpline):
    def __init__(self, x, y, period=2 * np.pi, sfactor=0, k=3, ext=0):
        """
        A PCSJ spline class
        Function to define a periodic spline that approximates a
        continous function sampled by x and y data points. If the repeat
        data point is not provided, it will be added to ensure a
        periodic trajectory """

        # Process inputs
        assert len(x) == len(y), "Length Mismatch"
        assert x.ndim == 1 & y.ndim == 1, "Too many dimensions"
        if not np.abs(x[-1] - period) < 1E-10:
            assert x[-1] < period, 'Data longer than 1 period'
            x = np.hstack([x, x[0] + period])
            y = np.hstack([y, y[0]])

        self.T = period
        self.ext = ext

        tck = splrep(x, y, s=sfactor, per=True, k=k)
        t, c, k = tck
        self._eval_args = tck
        self._data = (None, None, None, None, None, k, None, len(t), t,
                      c, None, None, None, None)

    def __call__(self, x, nu=0):
        return UnivariateSpline.__call__(self, x % self.T, nu=nu)

    def derivative(self, n=1):
        tck = fitpack.splder(self._eval_args, n)
        return PeriodicSpline._from_tck(tck, self.T)

    def antiderivative(self, n=1):
        tck = fitpack.splantider(self._eval_args, n)
        return PeriodicSpline._from_tck(tck)

    def root_offset(self, root=0):
        """ Return the values where the spline equals 'root'
        Restriction: only cubic splines are supported by fitpack.
        """
        t, c, k = self._eval_args
        new_c = np.array(c)
        new_c[np.nonzero(new_c)] += -root
        if k == 3:
            z, m, ier = dfitpack.sproot(t, new_c)
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]
        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')

    def integrate(self, a=0., b=2 * np.pi):
        """ Find the definite integral of the spline from a to b """

        # Are both a and b in (0, 2pi)?
        if (0 <= a <= 2 * np.pi) and (0 <= b <= 2 * np.pi):
            return splint(a, b, self._eval_args)
        elif ((a <= 0) and (b <= 0)) or ((a >= 2 * np.pi)
                                         and (b >= 2 * np.pi)):
            return splint(a % (2 * np.pi), b % (2 * np.pi), self._eval_args)

        elif (a <= 0) or (b >= 2 * np.pi):
            int = 0
            int += splint(a % (2 * np.pi), 2 * np.pi, self._eval_args)
            int += splint(0, b % (2 * np.pi), self._eval_args)
            return int

    @classmethod
    def _from_tck(cls, tck, period=2 * np.pi):
        """Construct a spline object from given tck"""
        self = cls.__new__(cls)
        self.T = period
        t, c, k = tck
        self._eval_args = tck
        #_data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        self._data = (None, None, None, None, None, k, None, len(t), t,
                      c, None, None, None, None)
        self.ext = 0
        return self


class ComplexPeriodicSpline:
    def __init__(self, x, y, period=2 * np.pi, sfactor=0):
        """
        A PCSJ spline class
        Class for complex periodic functions that will create two
        PeriodicSpline instances, one for real and one for imaginary
        components """

        yreal = np.real(y)
        yimag = np.imag(y)

        self.real_interp = PeriodicSpline(x, yreal, period, sfactor)
        self.imag_interp = PeriodicSpline(x, yimag, period, sfactor)

    def __call__(self, x, d=0):
        return self.real_interp(x, d) + 1j * self.imag_interp(x, d)

    def integrate(self, a, b):
        return (self.real_interp.integrate(a, b) +
                self.imag_interp.integrate(a, b) * 1j)


class MultivariatePeriodicSpline(object):
    def __init__(self, x, ys, period=2 * np.pi, sfactor=0, k=3):
        """
        A PCSJ spline class
        Combination class that supports a multi-dimensional input,
        will determine whether complex or regular periodic splines are
        needed. """

        self.iscomplex = np.any(np.iscomplex(ys))
        splinefn = (ComplexPeriodicSpline if self.iscomplex else
                    PeriodicSpline)

        self.splines = fnlist([])
        for y in np.atleast_2d(ys):
            y = y.squeeze()
            self.splines += [splinefn(x, y, period, sfactor, k)]

    def __call__(self, x, d=0):
        return self.splines(x, d).T

    def integrate(self, a=0, b=2 * np.pi):
        return np.array([interp.integrate(a, b) for interp in
                         self.splines])


class RootFindingSpline(InterpolatedUnivariateSpline):
    def root_offset(self, root=0):
        """ Return the values where the spline equals 'root'
        Restriction: only cubic splines are supported by fitpack.
        """
        t, c, k = self._eval_args
        new_c = np.array(c)
        new_c[np.nonzero(new_c)] += -root
        if k == 3:
            z, m, ier = dfitpack.sproot(t, new_c)
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]
        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')


def pca(data, n_components=None):
    """ returns a sklearn.decomposition.pca.PCA object fit to the data """
    pca = decomposition.pca.PCA(n_components=n_components)
    pca.fit(data)
    return pca


def save_nparray(filename, nparray, colnames=None):
    """
    Uses pandas to save a numpy array with column headers.
    """
    assert(len(colnames))==nparray.shape[1], "columns do not match table"
    output_df = pd.DataFrame(data=nparray, columns=colnames)
    output_df.to_csv(filename, index=False)

def save_inference_table(filename, table):
    """helper function for saving inference tables"""
    save_nparray(filename, table, colnames=['case_id','t','p_y','y'])

if __name__ == "__main__":

    # test roots
    times = np.arange(0, 10, 0.1)
    xvals = np.sin(times)
    sine_roots = roots(xvals, times=times)
    print('The roots of sine are:')
    print(sine_roots)
    print('Root finding successful.')
