import glob

import numpy as np
import pandas as pd

from datetime import datetime
from netCDF4 import Dataset
from verification_metrics import find_ice_edge, ice_edge_length
from scipy.interpolate import NearestNDInterpolator


def main():
    path_amsr2 = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2/"
    PATH_OUTPUT = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/"



    paths = []
    for month in range(1, 13):
        p = f"{path_amsr2}{2022}/{month:02d}/"
        paths.extend(sorted(glob.glob(f"{p}*")))

    outputs = []
    dates = []


    with Dataset(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_commons.nc") as constants:
        lsmask = constants['lsmask'][:,:]

    # baltic_mask = np.zeros_like(lsmask)
    # mask = np.zeros_like(lsmask)
    # baltic_mask[:1200, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    # mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    # mask = np.where(lsmask == 0)
    # mask_T = np.transpose(mask)

    for i, file in enumerate(paths):
        print(f"{i}/{len(paths) - 1}", end="\n")
        yyyymmdd = datetime.strptime(file[-11:-3], '%Y%m%d')

        with Dataset(file, 'r') as infile:
            sic = infile.variables['sic'][:, :]

            # sic_interpolator = NearestNDInterpolator(mask_T, sic[mask])
            # sic_processed = sic_interpolator(*np.indices(sic.shape))

            # lsmask = np.where(sic_processed == 100, 1, 0)

            ice_edge = find_ice_edge(sic, lsmask, threshold=2)
            outputs.append(ice_edge_length(ice_edge, s = 6.25))
            dates.append(yyyymmdd.strftime("%m-%d"))

    df_out = pd.DataFrame(columns = ['10-40%'], data=outputs, index = dates)
    df_out.index.name = 'date'
    df_out.to_csv(f"{PATH_OUTPUT}amsr2_ice_edge.csv")

    '''
    test_sics = np.empty((1, 368, 366))
    test_sics[0] = sic

    test_lsmask = np.empty_like(test_sics)
    test_lsmask[0] = lsmask

    with Dataset(f"{PATH_OUTPUT}amsr2_test.nc", 'w') as out:
        out.createDimension('x', 366)
        out.createDimension('y', 368)
        out.createDimension('t', 1)

        sic_out = out.createVariable('sic', 'd', ('t', 'y', 'x'))
        sic_out[:] = test_sics

        lsmask_out = out.createVariable('lsmask', 'd', ('t', 'y', 'x'))
        lsmask_out[:] = test_lsmask
    '''

if __name__ == "__main__":
    main()