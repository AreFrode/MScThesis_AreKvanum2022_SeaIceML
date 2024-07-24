import glob

import numpy as np
import pandas as pd

from datetime import datetime
from netCDF4 import Dataset
from verification_metrics import find_ice_edge_from_fraction, ice_edge_length
from scipy.interpolate import NearestNDInterpolator


def main():
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    path_icecharts = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/"



    paths = []
    for month in range(1, 13):
        p = f"{path_icecharts}{2022}/{month:02d}/"
        paths.extend(sorted(glob.glob(f"{p}*")))

    outputs = []
    dates = []


    with Dataset(f"{path_arome}2019/01/AROME_1kmgrid_20190101T18Z.nc") as constants:
        lsmask = constants['lsmask'][:,:-1]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:1200, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)

    for i, file in enumerate(paths):
        print(f"{i}/{len(paths) - 1}", end="\n")
        yyyymmdd = datetime.strptime(file[115:-9], '%Y%m%d')

        with Dataset(file, 'r') as infile:
            sic = infile.variables['sic'][:, :-1]
            sic_interpolator = NearestNDInterpolator(mask_T, sic[mask])
            sic_processed = sic_interpolator(*np.indices(sic.shape))
            sic_processed = sic_processed[578: ,:1792]

            lsmask = np.where(sic_processed == 100, 1, 0)

            ice_edge = find_ice_edge_from_fraction(sic_processed, lsmask, threshold=25)
            outputs.append(ice_edge_length(ice_edge, s = 1))
            dates.append(yyyymmdd.strftime("%m-%d"))

    df_out = pd.DataFrame(columns = ['10-40%'], data=outputs, index = dates)
    df_out.index.name = 'date'
    df_out.to_csv(f"{PATH_OUTPUT}icechart_ice_edge.csv")

    '''
    test_sics = np.empty((1, 1792, 1792))
    test_sics[0] = sic_processed

    with Dataset(f"{PATH_OUTPUT}icechart_test.nc", 'w') as out:
        out.createDimension('x', 1792)
        out.createDimension('y', 1792)
        out.createDimension('t', 1)

        sic_out = out.createVariable('sic', 'd', ('t', 'y', 'x'))
        sic_out[:] = test_sics
    '''


if __name__ == "__main__":
    main()