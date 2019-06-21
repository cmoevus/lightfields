import math
import os
from scipy import integrate
import pandas as pd
import scipy as sp
import functools
import warnings
from light_fields import constants, led
#warnings.filterwarnings('ignore', category=sp.integrate.IntegrationWarning)

#
# Constants
#
sun_ppfd = 2200  # PPFD of the Sun, the one we want to mimick
shelf_surface = 0.51 * 0.72  # The surface to cover at sun PPFD
cost_kwh = 0.26  # Cost of electricity per kWh, considering both supply and delivery charges...
h_light_per_day = 10  # Number of hours per day the LEDs will be turned on
shelf_heights = [0.90, 0.60] # The distance between the bottom and top of the shelves (ignoring pots)

#
# Calculation of costs and quantities
#
def surface_per_led(led_ppf):
    """Calculate the surface (cm2) a single LED covers to produce a PPFD similar to sunlight."""
    return led_ppf / sun_ppfd * 100**2

def leds_per_shelf(led_ppf):
    """Calculate the number of LEDs required to produce sun-like light on one shelf."""
    led_surface = led_ppf / sun_ppfd
    return np.ceil(shelf_surface / led_surface)

def monthly_cost_per_shelf(led_ppf, led_w):
    """Calculate the monthly cost of operating one shelf with sun-like light."""
    shelf_kwh = leds_per_shelf(led_ppf) * led_w / 1000
    return shelf_kwh * h_light_per_day * 365 * cost_kwh / 12

#
# Tools for approximating the ppf of a light source given in lumens
#
h = 6.626e-34
c = 3.0e+8
k = 1.38e-23
nA = 6.022140857e23

def gauss(x, A, mu, sigma):
    """Calculate a gaussian"""
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def luminosity_function(l):
    """Return the lm/W for the given wavelength"""
    return gauss(l, 683, 559, 42)

#
# Black body light sources
#
def planck(wav, T):
    """
    Calculate the Planck Law for the given temperature.

    Stolen from https://stackoverflow.com/questions/22417484/plancks-formula-for-blackbody-spectrum
    """
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

def eta_mu(T):
    """Calculate the lumens per PAR watts of a black body source of temperature T."""
    def a(l):
        return planck(l, T) * luminosity_function(l*10**9)
    b = sp.integrate.quad(planck, 400e-9, 700e-9, args=T)[0]
    return sp.integrate.quad(a, 400e-9, 700e-9)[0] / b

def eta_photon(T):
    """Calculate the mols of photon per second per PAR watts of a black body source of temperature T."""
    def a(l):
        return planck(l, T) * l / (h * c * nA)
    b = sp.integrate.quad(planck, 400e-9, 700e-9, args=T)[0]
    return sp.integrate.quad(a, 400e-9, 700e-9)[0] / b

def lumen_to_ppf(l, T):
    """Calculate the approximate ppf (in umol/s) of a light source of color temperature T emiting l lumen."""
    return l / eta_mu(T) * eta_photon(T) * 1e6

#
# Any light sources
#
def eta_mu_led(T, f):
    """Calculate the lumens per PAR watts of a LED with spectral distribution f at T °K of temperature T."""
    def a(l):
        return f(l) * luminosity_function(l)
    b = sp.integrate.quad(f, 400, 700, limit=500)[0]
    return sp.integrate.quad(a, 400, 700, limit=500)[0] / b

def eta_photon_led(T, f):
    """Calculate the mols of photon per second per PAR watts of a black body source of temperature T."""
    def a(l):
        return f(l) * l*1e-9 / (h * c * nA)
    b = sp.integrate.quad(f, 400, 700, limit=500)[0]
    return sp.integrate.quad(a, 400, 700, limit=500)[0] / b

def lumen_to_ppf_led(l, T, f):
    """Calculate the approximate ppf (in umol/s) of a light source of color temperature T emiting l lumen."""
    return l / eta_mu_led(T, f) * eta_photon_led(T, f) * 1e6

#
# Better functions for any light source
#
#def read_spectrum(f):
#    spectrum = pd.read_csv('datasets/wavelength_spectrums/' + f)
#    return led.estimate_spectrum(spectrum.iloc[:, 0], spectrum.iloc[:, 1])
def read_spectrum(f):
    """
    Return a list of interpolated graph from a spectral distribution file.
    It assumes the graph has been exported from WebPlotDigitizer and named as "CCT" or "CCT CRI", such as "3000K" or "3000K 70CRI".
    """
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
    dists = dict()
    for cct in ccts:
        data = df.loc[:, cct]
        func = sp.interpolate.interp1d(data['X'], data['Y'], assume_sorted=True)
        if 'CRI' in cct:
            cct, cri = cct.split(' ')
            cct = int(cct[:-1])
            cri = int(cri[:-3])
            if cct not in dists:
                dists[cct] = dict()
            dists[cct][cri] = func
        else:
            cct = int(cct[:-1]) if cct[:-1].isdigit() else cct
            dists[cct] = func
    return dists

def lumen_to_ppf_from_dist(dist_f, l, cct, cri=None, loose=True):
    """
    Estimate PPF (um/s) from the given spectral distribution, lumens, CCT and CRI.

    The loose option tolerates imprecisions in CRI and CCT.
    """
    # Get the spectral distribution
    path = os.path.join(
        constants.WAVELENGTH_SPECTRUM_DIRECTORY,
        dist_f
    )
    spectrum = read_spectrum(path)
    if cct in spectrum:
        dists = spectrum[cct]
    elif loose == True:
        dists = spectrum[list(spectrum.keys())[np.argmin(np.abs([i - cct for i in spectrum.keys()]))]]
    else:
        raise ValueError('Unknown spectrum for this color temperature')
    if type(dists) != dict:
        dist = dists
    elif (len(dists) == 1 and cri == None and loose == False) or (len(dists) == 1 and loose == True):
        dist = dists[list(dists.keys())[0]]
    elif cri in dists.keys():
        dist = dists[cri]
    else:
        raise ValueError('CRI needs to be defined.')

    # Calculate the value
    eta_mu = sp.integrate.quad(lambda l: dist(l) * luminosity_function(l), 400, 700, limit=500)[0] / sp.integrate.quad(dist, 400, 700, limit=500)[0]
    eta_photon = sp.integrate.quad(lambda l: dist(l) * l*1e-9 / (h * c * nA), 400, 700, limit=500)[0] / sp.integrate.quad(dist, 400, 700, limit=500)[0]
    return l / eta_mu * eta_photon * 1e6

leds = pd.read_csv('datasets/LEDs.csv', comment='#')

for i, p in enumerate(leds['ppf'].isnull()):
    if p == True:
        leds.set_value(i, 'ppf',
                       lumen_to_ppf_from_dist(leds.iloc[i]['spectrum'],
                                              leds.iloc[i]['lumens'],
                                              leds.iloc[i]['temperature'],
                                              leds.iloc[i]['cri']))

leds.set_index(['brand', 'series','model'], inplace=True)
leds.sort_index(inplace=True)
