"""
Various mathmatical functions
"""

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
