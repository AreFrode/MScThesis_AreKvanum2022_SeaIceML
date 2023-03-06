import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts")

import glob
import os

import numpy as np
import pandas as pd

from netCDF4 import Dataset
from tqdm import tqdm
from verification_metrics import sea_ice_extent, IIEE_alt, find_ice_edge, ice_edge_length

from scipy.interpolate import NearestNDInterpolator
from createHDF import onehot_encode_sic
from datetime import datetime
from dateutil.relativedelta import relativedelta
from loadClimatologicalIceEdge import load_climatological_ice_edge


def merge_to_df():
    PATH_ICECHARTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/"
    PATH_AROME = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    OUT_PATH = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/IceChartYears/"

    with Dataset(f"{PATH_AROME}2019/01/AROME_1kmgrid_20190101T18Z.nc", 'r') as constants:
        lsmask = constants['lsmask'][:]

    baltic_mask = np.zeros_like(lsmask)
    baltic_mask[:1200, 1500:] = 1
    mask = np.zeros_like(lsmask)

    mask = np.where(~np.logical_or((lsmask == 1), (baltic_mask == 1)))
    mask_T = np.transpose(mask)
    
    paths = []

    for year in range(2016, 2023):
        for month in range(1, 13):
            paths.append(f"{PATH_ICECHARTS}{year}/{month:02d}/")

    path_data_task = paths[int(sys.argv[1]) - 1]
    year_task = path_data_task[len(PATH_ICECHARTS) : len(PATH_ICECHARTS) + 4]
    month_task = path_data_task[len(PATH_ICECHARTS) + 5 : len(PATH_ICECHARTS) + 7]

    print(year_task)
    print(month_task)

    paths_task = sorted(glob.glob(f"{PATH_ICECHARTS}{year_task}/{month_task}/*.nc"))
    print(paths_task)
    print(len(paths_task))

    if int(sys.argv[1]) != 84:
        date = datetime.strptime(year_task+month_task,'%Y%m')
        next_month = datetime.strftime(date + relativedelta(months=+1), format="%Y%m")

        paths_task.extend(sorted(glob.glob(f"{PATH_ICECHARTS}{next_month[:4]}/{next_month[4:]}/*.nc"))[:2])

        print(paths_task)
        print(len(paths_task))

    rows = []
    
    # for path in tqdm(paths[:20]):
    for path_bulletin in tqdm(paths_task[:-2]):
        yyyymmdd_bulletin = path_bulletin[-17:-9]
        print(yyyymmdd_bulletin)

        yyyymmdd_bulletin_dt = datetime.strptime(yyyymmdd_bulletin, '%Y%m%d')
        yyyymmdd_valid_dt = yyyymmdd_bulletin_dt + relativedelta(days=+2)
        yyyymmdd_valid = datetime.strftime(yyyymmdd_valid_dt, '%Y%m%d')

        with Dataset(path_bulletin, 'r') as nc_bul:
            sic_bulletin = onehot_encode_sic(nc_bul.variables['sic'][:])

        sic_bulletin_interpolator = NearestNDInterpolator(mask_T, sic_bulletin[mask])
        sic_bulletin = sic_bulletin_interpolator(*np.indices(sic_bulletin.shape))

        extent = sea_ice_extent(sic_bulletin, lsmask, threshold = 2, side_length = 1)
        niiee = np.nan

        valid_path = f"{PATH_ICECHARTS}{yyyymmdd_valid[:4]}/{yyyymmdd_valid[4:6]}/ICECHART_1kmAromeGrid_{yyyymmdd_valid}T1500Z.nc"
        if valid_path in paths_task:
            
            bulletin_ice_edge = find_ice_edge(sic_bulletin, lsmask, threshold = 2)
            bulletin_ice_edge_length = ice_edge_length(bulletin_ice_edge, s = 1)

            with Dataset(valid_path, 'r') as nc_val:
                sic_valid = onehot_encode_sic(nc_val.variables['sic'][:])

            sic_valid_interpolator = NearestNDInterpolator(mask_T, sic_valid[mask])
            sic_valid = sic_valid_interpolator(*np.indices(sic_valid.shape))

            valid_ice_edge = find_ice_edge(sic_valid, lsmask, threshold = 2)
            valid_ice_edge_length = ice_edge_length(valid_ice_edge, s = 1)

            iiee = IIEE_alt(sic_bulletin, sic_valid, lsmask, side_length = 1, threshold = 2)
            niiee =  (2*(iiee[0].sum() + iiee[1].sum())) / (bulletin_ice_edge_length + valid_ice_edge_length)

        rows.append([yyyymmdd_bulletin, extent, niiee])
        

    df = pd.DataFrame(columns = ['date', 'extent', 'niiee'], data = rows)
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")

    df.set_index('date', inplace = True)
    if not os.path.exists(f"{OUT_PATH}{year_task}/{month_task}/"):
        os.makedirs(f"{OUT_PATH}{year_task}/{month_task}/")

    df.to_csv(f"{OUT_PATH}{year_task}/{month_task}/persistence_extent_niiee.csv")
    

def main():
    merge_to_df()



if __name__ == "__main__":
    main()