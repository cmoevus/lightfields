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
