import os

from light_fields import constants
from light_fields.spectral_distribution import SpectralDistribution


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
        path = os.path.join(
            constants.WAVELENGTH_SPECTRUM_DIRECTORY,
            description['spectrum']
        )

        self.spectrum = SpectralDistribution(path)
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
