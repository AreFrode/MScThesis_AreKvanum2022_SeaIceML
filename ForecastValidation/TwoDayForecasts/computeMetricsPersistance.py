import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset')

import os
import pandas as pd
import glob
import h5py

import numpy as np

from verification_metrics import IIEE, find_ice_edge, ice_edge_length, contourAreaDistribution
from tqdm import tqdm
from datetime import datetime, timedelta

def main():
    PATH_PERSISTANCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/Data/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    icecharts = sorted(glob.glob(f"{PATH_PERSISTANCE}2022/**/*.hdf5"))

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask = constants['lsmask'][578:, :1792]

    output_list = []


    for icechart in tqdm(icecharts):
        yyyymmdd = icechart[-13:-5]

        yyyymmdd_target = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_target = (yyyymmdd_target + timedelta(days = 2)).strftime('%Y%m%d')

        with h5py.File(icechart, 'r') as infile:
            sic_target = infile['sic_target'][578:, :1792]
            sic_persistance = infile['sic'][578:, :1792]

        ice_edge_target = find_ice_edge(sic_target, lsmask)
        target_length = ice_edge_length(ice_edge_target)

        ice_edge_forecast = find_ice_edge(sic_persistance, lsmask)
        forecast_length = ice_edge_length(ice_edge_forecast)

        iiee = IIEE(sic_persistance, sic_target, lsmask)
        a_plus = iiee[0].sum()
        a_minus = iiee[1].sum()

        area_dist_target = contourAreaDistribution(sic_target, lsmask)
        area_dist_forecast = contourAreaDistribution(sic_persistance, lsmask)

        output_list.append([pd.to_datetime(yyyymmdd_target, format="%Y%m%d"), target_length, forecast_length, np.mean([target_length, forecast_length]), a_plus + a_minus, a_plus, a_minus] + area_dist_target.tolist() + area_dist_forecast.tolist())


    output_df = pd.DataFrame(output_list, columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus', 'target_area0', 'target_area1', 'target_area2', 'target_area3', 'target_area4', 'target_area5', 'target_area6', 'forecast_area0', 'forecast_area1', 'forecast_area2', 'forecast_area3', 'forecast_area4', 'forecast_area5', 'forecast_area6'])
    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}persistance.csv")
    


if __name__ == "__main__":
    main()