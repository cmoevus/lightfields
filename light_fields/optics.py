import os
import pandas as pd
import scipy as sp

from light_fields import constants

class Optics(object):
    """Define an optical component to associate with a LED, or the radiation distribution of the LED itself.

    Parameters:
    ----------
    definition: pandas.Series
        The optics database entry of the piece.

    """

    def __init__(self, description):
        """Instanciate the Reflector."""
        self.description = description
        self.path = os.path.join(
            constants.AXIAL_RADIATION_DISTRIBUTION_DIRECTORY,
            description['file']
        )

    def estimate(self, x=None, y=None, fit=False):
        """Interpolate the axial radiation distribution from a discretized distribution.

        Parameters:
        -----------
        x: list or str
            The values in x (angle in degrees). If a string is given, it will assume it's a path to a CSV file with columns x, y (no header), and will read it. If None, will read from description.
        y: list
            The values in y (user defined units)
        fit: bool or function
            If not false, will fit the distribution with the given function. If no function is given (True is given), it will use the lambertian function.

        """
        if x is None:
            x = self.path
        if type(x) is str:
            dist = pd.read_csv(x, header=None)
            dist.columns = ['x', 'y']
            self.path = x
        else:
            dist = pd.DataFrame(np.array([x, y]).T)
            dist.columns = ['x', 'y']

        # Add limits if need be
        if dist['x'].min() > -90:
            dist = dist.append({'x': -90, 'y': 0}, ignore_index=True)
        if dist['x'].max() < 90:
            dist = dist.append({'x': 90, 'y': 0}, ignore_index=True)
        dist.sort_values('x', inplace=True)
        equation = sp.interpolate.UnivariateSpline(dist['x'], dist['y'], k=3, s=50, ext=1)

        # If the user rather wants a less precise, but faster, fitting, here it is
        if fit is not False:
            # The default function is a lambertian
            if fit is True:
                fit = lambertian
            fitted = sp.optimize.curve_fit(fit, dist['x'], dist['y'])  # , p0=(dist(0), 10, dist(0) / 20, 50)
            def equation(x):
                return fit(x, *fitted[0])

        self.ard = equation

    def integral_ard(self):
        """Return the area under the axial radiation distribution."""
        if isinstance(self.ard, sp.interpolate.UnivariateSpline):
            return self.ard.integral(-90, 90)
        else:
            return sp.integrate.quad(self.ard, -90, 90)[0]
