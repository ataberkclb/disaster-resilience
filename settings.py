import os
from pathlib import Path

import numpy as np
import pandas as pd

# code from Bart Meyers
colors = ['red', 'dodgerblue', 'blueviolet', 'mediumseagreen', 'deeppink', 'coral', 'royalblue', 'midnightblue',
          'yellowgreen', 'darkgreen', 'mediumblue', 'DarkOrange', 'green', 'MediumVioletRed',
          'darkcyan', 'orangered', 'purple', 'cornflowerblue', 'saddlebrown', 'indianred', 'fuchsia', 'DarkViolet',
          'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
          'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'black', 'grey',
          'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellow'] * 100
# colors = sns.color_palette("Paired", n_colors=100)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

ROOT_DIR = Path(__file__).parent

BS_PATH = os.path.join(ROOT_DIR, "data", "antennas.json")

# Average height of buildings in an area (used for RMa 5G NR only)
AVG_BUILDING_HEIGHT = 15  # current number based on average two-story building
AVG_STREET_WIDTH = 10

# BASE STATION PROPERTIES
BS_RANGE = 5000  # maximum range of base stations based on the fact that UMa and UMi models cannot exceed 5km

# User equipment properties
UE_HEIGHT = 1.5  # height in meters

# to calculate the noise power
BOLTZMANN = 1.38e-23
TEMPERATURE = 283.15

VERTICAL_BORE = np.radians(8)  # degrees

VERTICAL_BEAMWIDTH3DB = np.radians(65)
HORIZONTAL_BEAMWIDTH3DB = 65

MINIMUM_SNR = 5  # - math.inf # 5 # dB
N_SECTOR = 3  # array antennas (number of sectors)

CUTOFF_VALUE_INTERFERENCE = 3  # the x highest signal BSs will not interfere.
poweramp_eff = 0.12  # 12.8
POWER_PERCENTAGE = 0.9 * poweramp_eff
PRECISION_MARGIN = 10e-6  # Margin used in the calculations of FDP and FSP
# ASSUMPTION maybe change the power percentage!
SLEEP_POWER = 75  # in Watt, power consumption of BS when in the sleep mode
# constant power consumption values0
POWER_DSP = 100 * N_SECTOR # in Watt, digital signal processor unit
POWER_AIRCOND = 225  # in Watt
MICROWAVELINK_POWER = 80  # in Watt
POWER_RECT = 100 * N_SECTOR # in Watt, rectifier
POWER_TRANSCEIVER = 100 * N_SECTOR



