

import numpy as np

from netCDF4 import Dataset

def main():
    path_arome = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    path_icecharts = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"

    sics = []
    edges = []
    outputs = []
    dates = []


    paths = []
    for month in range(1, 13):
        p = f"{path_icecharts}{2022}/{month:02d}/"
        paths.append(p)

    with Dataset(f"{path_arome}2019/01/AROME_1kmgrid_20190101T18Z.nc") as constants:
        lsmask = constants['lsmask'][:,:-1]

    baltic_mask = np.zeros_like(lsmask)
    mask = np.zeros_like(lsmask)
    baltic_mask[:1200, 1500:] = 1   # Mask out baltic sea, return only water after interp
    
    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)

    print(paths)


if __name__ == "__main__":
    main()