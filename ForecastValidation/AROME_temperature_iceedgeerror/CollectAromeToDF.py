import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")

import glob
import os

import numpy as np
import pandas as pd

from netCDF4 import Dataset
from tqdm import tqdm
from verification_metrics import IIEE_alt, find_ice_edge, ice_edge_length

from scipy.interpolate import NearestNDInterpolator
from common_functions import onehot_encode_sic
from datetime import datetime
from dateutil.relativedelta import relativedelta
from loadClimatologicalIceEdge import load_climatological_ice_edge




def merge_to_df():
    PATH_AROME = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/"
    OUT_PATH = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/AROME_temperature_iceedgeerror/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_2/"

    concentration = '15%'
    climatological_ice_edge = load_climatological_ice_edge(2022, concentration, 0)

    baltic_mask = np.zeros((2370, 1845))
    baltic_mask[:1200, 1500:] = 1

    # print(np.unique(baltic_mask))
    # exit()
    paths = []

    for month in range(1, 13):
            paths.append(f"{PATH_AROME}2022/{month:02d}/")

    path_data_task = paths[int(sys.argv[1]) - 1]
    month_task = path_data_task[len(PATH_AROME) + 5 : len(PATH_AROME) + 7]

    print(month_task)

    paths_task = sorted(glob.glob(f"{PATH_AROME}2022/{month_task}/*.nc"))

    rows = []
    
    # for path in tqdm(paths[:20]):
    for path_bulletin in tqdm(paths_task):
        yyyymmdd_bulletin = path_bulletin[-15:-7]
        print(yyyymmdd_bulletin)

        yyyymmdd_bulletin_dt = datetime.strptime(yyyymmdd_bulletin, '%Y%m%d')
        yyyymmdd_valid_dt = yyyymmdd_bulletin_dt + relativedelta(days=+2)
        yyyymmdd_valid = datetime.strftime(yyyymmdd_valid_dt, '%Y%m%d')

        try:
            arome = glob.glob(path_bulletin)[0]
            ml = glob.glob(f"{PATH_DATA}{yyyymmdd_bulletin[:4]}/{yyyymmdd_bulletin[4:6]}/PreparedSample_v{yyyymmdd_valid}_b{yyyymmdd_bulletin}.hdf5")[0]

        except IndexError:
            continue

        with Dataset(arome, 'r') as nc_bul:
            sic_bulletin = onehot_encode_sic(np.nan_to_num(nc_bul.variables['sic'][:], nan=7.))

        # sic_bulletin = np.where(mask == 1, sic_bulletin, 0)
        mask = np.where(~np.logical_or((baltic_mask == 1), (sic_bulletin == -10)))
        mask_T = np.transpose(mask)
        sic_bulletin_interpolator = NearestNDInterpolator(mask_T, sic_bulletin[mask])
        sic_bulletin = sic_bulletin_interpolator(*np.indices(sic_bulletin.shape))

        with Dataset(ml, 'r') as nc:
            sic_target = nc.variables['sic'][578:, :1792]
            lsmask = nc.variables['lsmask'][578:, :1792]

        sic_bulletin = sic_bulletin[578:, :1792]


        # bulletin_ice_edge = find_ice_edge(sic_bulletin, lsmask, threshold = 2)
        # bulletin_ice_edge_length = ice_edge_length(bulletin_ice_edge, s = 1)


        # target_ice_edge = find_ice_edge(sic_target, lsmask, threshold = 2)
        # target_ice_edge_length = ice_edge_length(target_ice_edge, s = 1)

        iiee = IIEE_alt(sic_bulletin, sic_target, lsmask, side_length = 1, threshold = 2)
        niiee =  (2*(iiee[0].sum() + iiee[1].sum())) / climatological_ice_edge[concentration].loc[yyyymmdd_bulletin]

        rows.append([yyyymmdd_bulletin, niiee])
        

    df = pd.DataFrame(columns = ['date', 'niiee'], data = rows)
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")

    df.set_index('date', inplace = True)
    if not os.path.exists(f"{OUT_PATH}2022/{month_task}/"):
        os.makedirs(f"{OUT_PATH}2022/{month_task}/")

    df.to_csv(f"{OUT_PATH}2022/{month_task}/persistence_extent_niiee.csv")

    df.to_csv(f"{OUT_PATH}arome_icechart_niiee.csv")
    

def main():
    merge_to_df()



if __name__ == "__main__":
    main()