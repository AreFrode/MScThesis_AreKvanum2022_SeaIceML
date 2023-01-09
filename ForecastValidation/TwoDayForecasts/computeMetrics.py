import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET')

import os
import pandas as pd
import glob
import h5py

import numpy as np

from verification_metrics import IIEE, find_ice_edge, ice_edge_length, contourAreaDistribution, minimumDistanceToIceEdge
from tqdm import tqdm
from helper_functions import read_config_from_csv


def main():
    model_name = sys.argv[1]
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/{model_name}/"

    config = read_config_from_csv(f"{PATH_FORECAST[:-22]}configs/{model_name}.csv")

    # PATH_PERSISTANCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/osisaf_trend_{config['osisaf_trend']}/"


    PATH_OUTPUTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{config['lead_time']}/osisaf_trend_{config['osisaf_trend']}/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    icecharts = sorted(glob.glob(f"{PATH_PERSISTENCE}2022/**/*.hdf5"))
    forecasts = sorted(glob.glob(f"{PATH_FORECAST}2022/**/*.hdf5"))

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask = constants['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]
        lat = constants['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
        lon = constants['lon'][config['lower_boundary']:, :config['rightmost_boundary']]


    output_list = []

    for target, forecast in tqdm(zip(icecharts, forecasts), total = len(icecharts)):
        date = forecast[-17:-9]

        with h5py.File(target, 'r') as infile:
            sic_target = infile['sic_target'][config['lower_boundary']:, :config['rightmost_boundary']]

        with h5py.File(forecast, 'r') as infile:
            sic_forecast = infile['y_pred'][0]

        ice_edge_target = find_ice_edge(sic_target, lsmask)
        target_length = ice_edge_length(ice_edge_target)

        ice_edge_forecast = find_ice_edge(sic_forecast, lsmask)
        forecast_length = ice_edge_length(ice_edge_forecast)

        iiee = IIEE(sic_forecast, sic_target, lsmask)
        a_plus = iiee[0]
        a_minus = iiee[1]

        area_dist_target = contourAreaDistribution(sic_target, lsmask)
        area_dist_forecast = contourAreaDistribution(sic_forecast, lsmask)

        a_plus_minimum_distance = minimumDistanceToIceEdge(a_plus, ice_edge_target, lat, lon)
        a_minus_minimum_distance = minimumDistanceToIceEdge(a_minus, ice_edge_target, lat, lon)

        output_list.append([pd.to_datetime(date, format="%Y%m%d"), target_length, forecast_length, np.mean([target_length, forecast_length]), a_plus.sum() + a_minus.sum(), a_plus.sum(), a_minus.sum(), a_plus_minimum_distance.mean(), a_minus_minimum_distance.mean()] + area_dist_target.tolist() + area_dist_forecast.tolist())

    output_df = pd.DataFrame(output_list, columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus', 'mean_minimum_distance_to_ice_edge_a_plus', 'mean_minimum_distance_to_ice_edge_a_minus', 'target_area0', 'target_area1', 'target_area2', 'target_area3', 'target_area4', 'target_area5', 'target_area6', 'forecast_area0', 'forecast_area1', 'forecast_area2', 'forecast_area3', 'forecast_area4', 'forecast_area5', 'forecast_area6'])

    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}{model_name}.csv")
    


if __name__ == "__main__":
    main()