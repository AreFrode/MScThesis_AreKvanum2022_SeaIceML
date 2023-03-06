import glob
import os

import pandas as pd
import numpy as np

from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from verification_metrics import find_ice_edge_from_fraction, ice_edge_length
from datetime import datetime
from tqdm import tqdm

from matplotlib import pyplot as plt

def main():
    # path to data
    OSISAF_climatology_2011_2020 = glob.glob("/lustre/storeB/users/cyrilp/Data/OSI-SAF/SIP_OSISAF_climatology/OSISAF_climatology_SIP_2011_2020.nc")[0]
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/"

    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"
    proj4_osisaf = "+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=90.0"

    crs_AROME = CRS.from_proj4(proj4_arome)
    crs_OSISAF = CRS.from_proj4(proj4_osisaf)

    transform_function = Transformer.from_crs(crs_OSISAF, crs_AROME, always_xy = True)
    
    sics = []
    edges = []
    outputs = []
    dates = []

    # AROME grid
    x_min = 279103.2
    x_max = 2123103.2
    y_min = -897431.6
    y_max = 1471568.4

    step = 25000

    x_target = np.arange(x_min, x_max + step, step)
    y_target = np.arange(y_min, y_max + step, step)

    with Dataset(OSISAF_climatology_2011_2020, 'r') as f:
        x_input = f.variables['xc'][:]*1000
        y_input = f.variables['yc'][:]*1000
        lat_input = f.variables['lat'][:]
        lon_input = f.variables['lon'][:]

    xx_ic, yy_ic = np.meshgrid(x_input, y_input)

    xx_arome, yy_arome = transform_function.transform(xx_ic, yy_ic)
    xx_arome_flat = xx_arome.flatten()
    yy_arome_flat = yy_arome.flatten()

    baltic_mask = np.zeros((len(y_target), len(x_target)))
    baltic_mask[0:52, 50:len(x_target)] = 1

    with Dataset(OSISAF_climatology_2011_2020, 'r') as f:
        for day in tqdm(f.variables['time']):
            sic = f.variables['SIC'][int(day) - 1,:,:]
            sic_flat = sic.flatten()    
            sic_arome = griddata((xx_arome_flat, yy_arome_flat), sic_flat, (x_target[None, :], y_target[:, None]), method = 'nearest')
            sic_output = np.where(np.logical_and(baltic_mask == 1, ~np.isnan(sic_arome)) , 0, sic_arome)

            lsmask = np.where(np.isnan(sic_output), 1, 0)
            ice_edge = [find_ice_edge_from_fraction(sic_output, lsmask, threshold = i) for i in [5, 15, 25, 55, 80, 95]]
 
            sics.append(sic_output)
            edges.append(ice_edge)
            outputs.append([ice_edge_length(i, s = 25) for i in ice_edge])

            dates.append(datetime.strptime(f"2020{int(day):03d}", '%Y%j').strftime("%m-%d"))


    with Dataset(f"{PATH_OUTPUT}test.nc", 'w') as out:
        out.createDimension('x', len(x_target))
        out.createDimension('y', len(y_target))
        out.createDimension('contour', 6)
        out.createDimension('t', 366)

        """
        latc = out.createVariable('lat', 'd', ('y', 'x'))
        latc.units = 'degrees_north'
        latc.standard_name = 'Latitude'
        latc[:] = lat_arome

        lonc = out.createVariable('lon', 'd', ('y', 'x'))
        lonc.units = 'degrees_east'
        lonc.standard_name = 'Longitude'
        lonc[:] = lon_arome
        """

        sic_out = out.createVariable('sic', 'd', ('t', 'y', 'x'))
        sic_out[:] = sics
 

        # bmask = out.createVariable('BalticMask', 'd', ('y', 'x'))
        # bmask[:] = baltic_mask

        iedge = out.createVariable('IceEdge', 'd', ('t', 'contour', 'y', 'x'))
        iedge[:] = edges

    df_out = pd.DataFrame(columns = ['5%', '15%', '25%', '55%', '80%', '95%'], data=outputs, index=dates)
    df_out.index.name = 'date'
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_out.to_csv(f"{PATH_OUTPUT}climatological_ice_edge.csv")


if __name__ == "__main__":
    main()