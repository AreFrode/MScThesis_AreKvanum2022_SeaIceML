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
from two_day_forecast.createHDF import onehot_encode_sic


def main():
    PATH_PERSISTANCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/Data/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    icecharts = sorted(glob.glob(f"{PATH_PERSISTANCE}2021/**/*.hdf5", recursive = True))
    

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask = constants['lsmask'][450:, :1840]

    output_df = pd.DataFrame(columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus', 'target_area0', 'target_area1', 'target_area2', 'target_area3', 'target_area4', 'target_area5', 'target_area6', 'forecast_area0', 'forecast_area1', 'forecast_area2', 'forecast_area3', 'forecast_area4', 'forecast_area5', 'forecast_area6'])

    for icechart in tqdm(icecharts):
        yyyymmdd = icechart[-13:-5]

        yyyymmdd_target = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_target = (yyyymmdd_target + timedelta(days = 2)).strftime('%Y%m%d')

        with h5py.File(icechart, 'r') as infile:
            sic_target = infile['sic_target'][450:, :1840]
            sic_persistance = onehot_encode_sic(infile['sic'][450:, :1840])

        ice_edge_target = find_ice_edge(sic_target, lsmask)
        target_length = ice_edge_length(ice_edge_target)

        ice_edge_forecast = find_ice_edge(sic_persistance, lsmask)
        forecast_length = ice_edge_length(ice_edge_forecast)

        iiee = IIEE(sic_persistance, sic_target, lsmask)
        a_plus = iiee[0].sum()
        a_minus = iiee[1].sum()

        area_dist_target = contourAreaDistribution(sic_target)
        area_dist_forecast = contourAreaDistribution(sic_persistance)

        df_column_values = [pd.to_datetime(yyyymmdd_target, format="%Y%m%d"), target_length, forecast_length, np.mean([target_length, forecast_length]), a_plus + a_minus, a_plus, a_minus] + area_dist_target.tolist() + area_dist_forecast.tolist()

        tmp_df = pd.DataFrame([df_column_values], columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus', 'target_area0', 'target_area1', 'target_area2', 'target_area3', 'target_area4', 'target_area5', 'target_area6', 'forecast_area0', 'forecast_area1', 'forecast_area2', 'forecast_area3', 'forecast_area4', 'forecast_area5', 'forecast_area6'])
        output_df = pd.concat([output_df, tmp_df])

    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}persistance.csv")
    


if __name__ == "__main__":
    main()