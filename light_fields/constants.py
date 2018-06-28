SUN_PPFD = 2200  # PPFD of the Sun, the one we want to mimick
SHELF_SURFACE = 0.51 * 0.72  # The surface to cover at sun PPFD
COST_KWH = 0.26  # Cost of electricity per kWh, considering both supply and delivery charges...
MOLES_PER_M2_PER_DAY = 50  # Number of moles of photon per meter square per day

WAVELENGTH_SPECTRUM_DIRECTORY = 'datasets/wavelength_spectrums'
AXIAL_RADIATION_DISTRIBUTION_DIRECTORY = 'datasets/axial_radiation_distributions'

# To use absolute path
#import os
#WAVELENGTH_SPECTRUM_DIRECTORY = os.path.abspath('datasets/wavelength_spectrums')
#AXIAL_RADIATION_DISTRIBUTION_DIRECTORY = os.path.abspath('datasets/axial_radiation_distributions')
