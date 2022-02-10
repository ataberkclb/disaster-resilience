from settings import *
import util
import json
import objects.BaseStation as BSO
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import pandas
import numpy as np
import matplotlib.pyplot as plt

# code from Bart Meyers

def find_zip_code_region(zip_codes, city=None):
    if city == None:
        city = list(zip_codes['municipali'])
    zip_code_region_data = zip_codes[zip_codes['municipali'].isin(city)]
    region = gpd.GeoSeries(unary_union(zip_code_region_data['geometry']))
    return region, zip_code_region_data

def load_bs(zip_code_region):
    all_basestations = list()
    with open(BS_PATH) as f:
        bss = json.load(f)
        # Loop over base-stations
        for bs in bss:
            x = float(bs.get('x'))
            y = float(bs.get('y'))
            if zip_code_region.contains(Point(x, y)).bool():
                if bs.get("type") == "LTE":
                    radio = util.BaseStationRadioType.LTE
                elif bs.get("type") == "5G NR":
                    radio = util.BaseStationRadioType.NR
                elif bs.get("type") == "GSM":
                    radio = util.BaseStationRadioType.GSM
                elif bs.get("type") == "UMTS":
                    radio = util.BaseStationRadioType.UMTS
                else:
                    print(bs.get("HOOFDSOORT")) # there are no other kinds of BSs in this data set (yet)

                h = bs.get('antennes')[0].get("height") # the height of all antenna's is the same for 1 BS.
                new_bs = BSO.BaseStation(bs.get('ID'), radio, x, y, h)
                for antenna in bs.get("antennes"):
                    frequency = util.str_to_float(antenna.get("frequency"))
                    power = util.str_to_float(antenna.get("power")) # in dBW?
                    angle = util.str_to_float(antenna.get('angle'))
                    new_bs.add_channel(frequency, power, angle)
                all_basestations.append(new_bs)

    return all_basestations



