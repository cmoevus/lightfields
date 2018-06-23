#!/bin/env python
"""A package for representing LEDs."""
import numpy as np
import pandas as pd
import scipy as sp
import scipy.constants
import scipy.interpolate
import os

OPTIONS = {
    'wavelength spectrum directory': '/home/corentin/Projets/Vertical Garden/Light/Wavelength spectrums',
    'axial radiation distribution directory': '/home/corentin/Projets/Vertical Garden/Light/Axial radiation distributions'
}


def luminosity_function(x):
    """Return the lm/W for the given wavelength.

    Note that this is an approximation, fitted by a Gaussian.

    """
    A = 683
    mu = 559
    sigma = 42
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def lambertian(x, I0, m):
    """Lambertian distribution function."""
    return I0 * np.cos(np.radians(x))**m


def two_lambertian(x, I1, m1, I2, m2):
    """Sum of two Lambertian distribution functions."""
    return I1 * np.cos(np.radians(x))**m1 + I2 * np.cos(np.radians(x))**m2


def split_spectrums(f):
    """Separates a file with multiples spectrums, as exported from WebPlotDigitizer into files with single spectrums.

    Parameters:
    -----------
    f: str
        The path to the spectrum file.

    """
    name = os.path.splitext(f)[0]
    df = pd.read_csv(f, header=[0, 1])

    # Clean up
    prev, remove, ccts = None, list(), list()
    for column in df:
        if 'Unnamed' in column[0]:
            remove.append(column)
            df.loc[:, (prev, column[1])] = df[column[0]][column[1]]
        else:
            prev = column[0]
            ccts.append(column[0])
    df.drop(remove, axis=1, inplace=True)

    # Make a distribution for each CCT
    for cct in ccts:
        data = df.loc[:, cct].dropna()

        # Add missing extrema?
        if np.min(data['X']) > 400:
            data.loc[-1] = [400, 0]
        if np.max(data['X']) < 800:
            data.loc[len(data)] = [800, 0]

        data.sort_index().to_csv(name + " " + cct + '.csv', index=False, header=False)

    os.remove(f)


class CSVDatabase(object):
    """Build a database-like structure from a CSV file.

    Parameters:
    -----------
    db_file: str or pd.DataFrame
        Path to the csv file containing the database, or pandas DataFrame of the database

    """

    def __init__(self, db_file=None):
        """Instanciate a CSV database."""
        if type(db_file) != pd.DataFrame:
            self.db_path = db_file
            if db_file is not None:
                self.load_db()
        else:
            self.db = db_file
            self.db_path = None

    def load_db(self, db_file=None):
        """Load a CSV file as its database."""
        if db_file is None:
            db_file = self.db_path
        self.db = pd.read_csv(db_file)

    def find(self, **kwargs):
        """Return a list of rows, from the database, that fit the given criteria.

        All given criteria have to be satisfied for a row to be returned.
        """
        expression = True
        for k, i in kwargs.items():
            ref = self.db[k]
            if type(i) == str:
                ref = ref.str.lower()
                i = i.lower()
            expression = np.logical_and(ref == i, expression)
        return self.db.where(expression).dropna(how='all').reset_index()

    def add(self, **kwargs):
        """Add a component to the database."""
        i = len(self.db)
        for k, v in kwargs.items():
            self.db.loc[i, k] = v
        self.db.to_csv(self.db_path, index=False)

    def __iter__(self):
        """Make the database iterable."""
        self.counter = 0
        return self

    def __next__(self):
        """Return next row in the database."""
        self.counter += 1
        try:
            return self.db.iloc[self.counter - 1]
        except IndexError:
            raise StopIteration

    def __len__(self):
        """Count the number of rows of the database."""
        return len(self.db)


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


class LED(object):
    """Define a LED.

    Parameters:
    -----------
    definition: pandas.Series
        The LED database entry of the LED.
    z: float
        The vertical distance, in meters, between the LED and the illuminated surface.
    center: 2-tuple of floats
        The distance (x, y), in meters, between the LED and the center of its illumination field, on the illuminated surface (as a way to define the angle of the LED)

    Notes:
    -------
    - All distance units are defined by the user. All that matters is that they are consistent with each other (don't imx meters and centimeters, for example)
    """

    def __init__(self, description, z, center=(0, 0)):
        """Instanciate the LED."""
        self.description = description
        self.spectrum = SpectralDistribution(os.path.join(OPTIONS['wavelength spectrum directory'], description['spectrum']))
        self.z, self.center = z, center

    @property
    def ppf(self):
        """Return the photosynthetic photon flux of the LED, in um/s.

         If this value is not set, it will estimate it based on its spectral distribution and its lumen rating.
        The calculation is stolen from https://en.wikipedia.org/wiki/Photosynthetically_active_radiation

        """
        if not hasattr(self, '_ppf'):
            self._ppf = self.description['lumens'] / self.spectrum.lumens_per_watts * self.spectrum.photons_per_watts * 1e6
        return self._ppf

    @ppf.setter
    def ppf(self, x):
        self._ppf = x

    @property
    def heatrate(self):
        """Return the amount of watts lost in heat."""
        return self.power - self.lumens / self.spectrum.lumens_per_watts

    def axial_radiation_distribution(self, theta):
        """Return the intensity value at a given angle.

        Parameters:
        -----------
        theta : float
            The angle, in degrees

        """
        return self.optics.ard(theta)

    ard = axial_radiation_distribution

    def axial_photon_flux_distribution(self, theta):
        """Return the photon flux (umol / s) as a function of the angle (degrees)."""
        return self.optics.ard(theta) / self.optics.integral_ard() * self.ppf

    def distal_radiation_distribution(self, x, y):
        """Return the value for the given distance from the center of the distribution.

        The units of the returned values are meaningless. Use `distal_flux_density_distribution` to get values in umol/s.m**2.

        Parameters:
        ------------
        x, y: float
            The distance, in meters, between the center of the LED (which is not the center of the illumation field) and the point of interest on the illuminated field.

        """
        # # Without the LED angle
        # d_u = np.sqrt(x**2 + y**2)
        # theta = np.degrees(np.arctan(d_u / z))
        # return self.ard(theta)

        # LC = np.sqrt(z**2 + center[0]**2 + center[1]**2)
        # LU = np.sqrt(z**2 + x**2 + y**2)
        # CU = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        # theta = np.degrees(np.arccos((LC**2 + LU**2 - CU**2) / (2 * LU * LC)))

        theta = self.coords_to_angle(x, y, self.z, self.center)
        return self.ard(theta)

    drd = distal_radiation_distribution

    def coords_to_angle(self, x, y):
        """Transform x, y coordinates, in meters, from the center of the LED, to the angle from the center of the illumation area defined in `LED.center`."""
        LC = np.sqrt(self.z**2 + self.center[0]**2 + self.center[1]**2)
        LU = np.sqrt(self.z**2 + x**2 + y**2)
        CU = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        theta = np.degrees(np.arccos((LC**2 + LU**2 - CU**2) / (2 * LU * LC)))
        return theta

    # def area_under_drd(self, z, center=(0, 0)):
    #     """Return the area under the curve of the distal radiation function for a given height.
    #
    #     Max precision in z: 3 decimals
    #
    #     """
    #     # z = round(z, 3)
    #     # if not hasattr(self, 'integral_drd'):
    #     #     self.integral_drd = dict()
    #     # if z not in [round(i, 3) for i in self.integral_drd.keys()]:
    #     #     self.integral_drd[z] = dict()
    #     # if center not in self.integral_drd[z].keys():
    #     #     self.integral_drd[z][center] = sp.integrate.nquad(self.distal_radiation_distribution, [[-10, 10], [-10, 10]], args=(z, center))[0]
    #     # return self.integral_drd[z][center]
    #     return sp.integrate.nquad(self.distal_radiation_distribution, [[-1000, 1000], [-1000, 1000]], args=(z, center))[0]

    def distal_photon_flux_density_distribution(self, x, y, precision=0.0001):
        """Return the flux density, in umol/(s * m**2) at a given position.

        Parameters:
        -----------
        x, y: float
            the coordinates, in meters, of the point of interest
        precision: float
            the precision, in meters, to use to calculate the density in the area of interest. The smallest the precision, the more exact the returned value.

        """
        # return self.optics.distal_radiation_distribution(x, y, self.z, self.center) * self.ppf / self.optics.area_under_drd(self.z, self.center)
        # Define the points surrounding the point of interest
        precision_top = x + precision, y + precision
        precision_bottom = x - precision, y - precision

        # Define the precision in degrees
        precision_deg = abs(self.coords_to_angle(*precision_top) - self.coords_to_angle(*precision_bottom))

        # Define the angle of interest
        theta = self.coords_to_angle(x, y)  # The angle from the center of the distribution

        # Define the umol/s in the area of precision
        ppf = sp.integrate.quad(self.axial_photon_flux_distribution, theta - precision_deg / 2, theta + precision_deg / 2)[0]

        # Define the area of precision
        dCT = np.sqrt((precision_top[0] - self.center[0])**2 + (precision_top[1] - self.center[1])**2)
        dCB = np.sqrt((precision_bottom[0] - self.center[0])**2 + (precision_bottom[1] - self.center[1])**2)
        area = np.pi * (dCT**2 - dCB**2)
        print(precision_top, precision_bottom, precision_deg, dCT, dCB, area, ppf)

        return ppf / area


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
        self.path = os.path.join(OPTIONS['axial radiation distribution directory'], description['file'])

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


class LightField(object):
    """Define a field of light with different LEDs at different positions.

    Parameters:
    -----------
    position: list
        a list of LED and their positions, with each entry in the format (x, y, z, (xc, yc), led).
        x, y, z, (xc, yc): floats
            The position of the LED, z being height from where the light field will be observed, and (xc, yc), the center of the light distribution at z relative to the center of the LED.
        led: LED object
            An LED object with its optics defined.

    """

    def __init__(self, positions):
        """Instanciate the instance."""
        self.positions = positions

    @property
    def z(self):
        """Define the distance between the point of observation and the lowest LED in the lightfield, keeping into consideration the vertical distance between the LEDs themselves."""
        raise NotImplemented

    @z.setter
    def z(self, z):
        raise NotImplemented

    def estimate(self, w, d, precision=0.01):
        """Calculate the light field.

        Parameters:
        ------------
        w: float
            the width of the light field, in m.
        d: float
            the height of the light field, in m.
        precision: float
            the number of meters per pixel.

        """
        field = np.zeros((int(d / precision), int(w / precision)))
        for x, y, z, center, led in self.positions:
            x_coordinates, y_coordinates = np.arange(0 - x, w - x, precision), np.arange(0 - y, d - y, precision)
            pos_x, pos_y = np.meshgrid(x_coordinates, y_coordinates)
            field += led.distal_flux_density_distribution(pos_x, pos_y, z, center)
        return field


if __name__ == "__main__":
    optics_db = CSVDatabase('Light/Optics.csv')
    led_db = CSVDatabase('Light/LEDs.csv')
