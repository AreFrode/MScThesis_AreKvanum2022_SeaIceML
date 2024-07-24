import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset')

import os
import pandas as pd
import glob
import h5py

import numpy as np

from verification_metrics import IIEE_alt, find_ice_edge_from_fraction, ice_edge_length, contourAreaDistribution, root_mean_square_error
from tqdm import tqdm
from datetime import datetime, timedelta
from loadClimatologicalIceEdge import load_climatological_ice_edge
from netCDF4 import Dataset
from calendar import monthrange

def main():
    lead_time = int(sys.argv[1])
    res = sys.argv[2]
    res_int = int(res[:-2])

    PATH_ICECHART = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/{res}/"
    PATH_OUTPUTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{lead_time}/"
    
    PATH_CONSTANTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Data/{res}/2022/01/AROME_{res}grid_20220101T18Z.nc"
    with Dataset(PATH_CONSTANTS, 'r') as constants:
        lsmask = constants['lsmask'][:]

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    years = [2019, 2020, 2021, 2022]
    months = []
    days = []

    for month in range(1, 13):
        months.append(month)
        days.append(monthrange(int(years[-1]), (month))[1])


    output_list = []

    # for i, month in enumerate(tqdm(months)):
        # for dd in tqdm(range(1, days[i] + 1), leave = False):
    # for year in tqdm(years):
        # for i, month in enumerate(tqdm(months, leave = False)):
    thread = int(sys.argv[3]) - 1
    year = years[int(thread/ 12)]
    month = months[thread % 12]

    for dd in range(1, days[thread % 12] + 1):

        yyyymmdd = f"{year}{month:02d}{dd:02d}"
        print(yyyymmdd)

        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = lead_time)).strftime('%Y%m%d')

        try:
            current_path = glob.glob(f"{PATH_ICECHART}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/ICECHART_{res}AromeGrid_{yyyymmdd}T1500Z.nc")[0]
            target_path = glob.glob(f"{PATH_ICECHART}{yyyymmdd_valid[:4]}/{yyyymmdd_valid[4:6]}/ICECHART_{res}AromeGrid_{yyyymmdd_valid}T1500Z.nc")[0]

        except IndexError:
            continue

        with Dataset(current_path, 'r') as infile:
            sic_start = infile['sic'][:]

        with Dataset(target_path, 'r') as infile:
            sic_target = infile['sic'][:]

        ice_edge_target = find_ice_edge_from_fraction(sic_target, lsmask, threshold = 25)
        target_length = ice_edge_length(ice_edge_target, s = res_int)

        ice_edge_forecast = find_ice_edge_from_fraction(sic_start, lsmask, threshold = 25)
        forecast_length = ice_edge_length(ice_edge_forecast, s = res_int)

        mean_length = 0.5*(target_length + forecast_length)

        NIIEE = []
        for i in [5, 25, 55, 80, 95, 100]:
            iiee = IIEE_alt(sic_start, sic_target, lsmask, side_length = res_int, threshold = i)
            a_plus = float(iiee[0].sum())
            a_minus = float(iiee[1].sum())
            NIIEE.append((a_plus + a_minus) / mean_length)


        output_list.append([pd.to_datetime(yyyymmdd, format="%Y%m%d"), target_length, forecast_length, mean_length, *NIIEE])


    output_df = pd.DataFrame(output_list, columns = ['date', 'target_length', 'forecast_length', 'mean_length', *[f"NIIEE_{i}" for i in range(1, 7)]])

    output_df = output_df.set_index('date')

    if not os.path.exists(f"{PATH_OUTPUTS}{year}/{month}/"):
        os.makedirs(f"{PATH_OUTPUTS}{year}/{month}/")

    output_df.to_csv(f"{PATH_OUTPUTS}{year}/{month}/persistence_{res}.csv")
    


if __name__ == "__main__":
    main()