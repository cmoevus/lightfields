import numpy as np
import pandas as pd
import scipy as sp
import scipy.interpolate


class SpectralDistribution(object):
    """Define a spectral distribution.

    The instance will behave as a function, such that you can call your instance to get a value:

    >>> sd = SpectralDistribution(path)
    >>> sd(self.min_x)
    1450
    >>> sd.lumens_per_watts
    3.45

    Calculations for the lumens_per_watts and photons_per_watts were stolen from https://en.wikipedia.org/wiki/Photosynthetically_active_radiation.

    Parameters:
    ----------
    path: str
        The path to the file describing the spectral distribution. See SpectralDistribution.estimate()
    equation: func
        An equation that takes the wavelength as input and returns the amount of photons, or relative intensity, or such
    boundaries: 2-tuple
        The limits in x where the distribution is above 0.

    """

    def __init__(self, path=None, equation=None, boundaries=(-np.inf, np.inf)):
        """Instanciate the object."""
        self.path = path
        self.equation = equation
        self.min_x, self.max_x = boundaries
        if path is not None:
            self.estimate(path)

    def __call__(self, x):
        """Return the value of the distribution at wavelength x (in nm)."""
        try:
            return self.equation(x)
        except TypeError:
            raise ValueError('The equation of the spectral distribution equation has not been defined.')

    def estimate(self, x, y=None):
        """Interpolate an equation from a discretized spectral distribution.

        Parameters:
        -----------
        path
        x: list or str
            The values in x (wavelength, nm). If a string is given, it will assume it's a path to a CSV file with columns x, y (no header), and will read it.
        y: list
            The values in y (user defined units)

        """
        # Load file
        if type(x) == str:
            spectrum = pd.read_csv(x, header=None)
            self.path = x
            x, y = spectrum.iloc[:, 0], spectrum.iloc[:, 1]

        # Interpolate and define boundaries to avoid errors...
        self.min_x, self.max_x = min(x), max(x)
        self.equation = sp.interpolate.UnivariateSpline(x, y, k=3, s=50, ext=1)

        # Delete previously calculated properties, so that they're actualized
        try:
            del self._lm_p_W
        except AttributeError:
            pass
        try:
            del self._mol_p_W
        except AttributeError:
            pass

    @property
    def lumens_per_watts(self):
        """Return the amount of lumens this source produces per watts of light (in lm/W)."""
        if not hasattr(self, '_lm_p_W'):
            self._lm_p_W = sp.integrate.quad(lambda l: self(l) * luminosity_function(l), self.min_x, self.max_x)[0] / self.equation.integral(self.min_x, self.max_x)
        return self._lm_p_W

    @property
    def photons_per_watts(self):
        """Return the amount of photons this source produces per watts of light (in mol/W)."""
        if not hasattr(self, '_mol_p_W'):
            self._mol_p_W = sp.integrate.quad(lambda l: self(l) * l * 1e-9 / (sp.constants.h * sp.constants.c * sp.constants.N_A), self.min_x, self.max_x)[0] / self.equation.integral(self.min_x, self.max_x)
        return self._mol_p_W
