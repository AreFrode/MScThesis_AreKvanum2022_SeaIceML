import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/')

import os
import pandas as pd
import glob
import h5py

import numpy as np

from verification_metrics import IIEE_alt, find_ice_edge, ice_edge_length, contourAreaDistribution, minimumDistanceToIceEdge, root_mean_square_error
from tqdm import tqdm
from tqdm.contrib import tzip
from helper_functions import read_config_from_csv
from loadClimatologicalIceEdge import load_climatological_ice_edge


def main():
    model_name = sys.argv[1]
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/Data/{model_name}/"

    config = read_config_from_csv(f"{PATH_FORECAST[:-22]}configs/{model_name}.csv")

    concentration = '15%'
    climatological_ice_edge = load_climatological_ice_edge(2022, concentration, int(config['lead_time']))

    # PATH_PERSISTANCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    if config['reduced_classes']:
        PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/reduced_classes/lead_time_{config['lead_time']}/"
    
    else:
        PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{config['lead_time']}/"

    PATH_OUTPUTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{config['lead_time']}/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    icecharts = sorted(glob.glob(f"{PATH_PERSISTENCE}2022/**/*.hdf5"))
    forecasts = sorted(glob.glob(f"{PATH_FORECAST}2022/**/*.hdf5"))

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask = constants['lsmask'][config['lower_boundary']:, :config['rightmost_boundary']]
        lat = constants['lat'][config['lower_boundary']:, :config['rightmost_boundary']]
        lon = constants['lon'][config['lower_boundary']:, :config['rightmost_boundary']]

    print(len(icecharts))
    print(len(forecasts))

    output_list = []

    for target, forecast in tzip(icecharts, forecasts):
        date = forecast[-17:-9]

        with h5py.File(target, 'r') as infile:
            sic_target = infile['sic_target'][config['lower_boundary']:, :config['rightmost_boundary']]

        with h5py.File(forecast, 'r') as infile:
            sic_forecast = infile['y_pred'][0]

        # rmse = root_mean_square_error(sic_forecast, sic_target, lsmask)

        ice_edge_target = find_ice_edge(sic_target, lsmask)
        target_length = ice_edge_length(ice_edge_target)

        ice_edge_forecast = find_ice_edge(sic_forecast, lsmask)
        forecast_length = ice_edge_length(ice_edge_forecast)
        
        NIIEE = []
        for i in range(1, config['num_outputs']):
            iiee = IIEE_alt(sic_forecast, sic_target, lsmask, side_length = 1, threshold = i)
            NIIEE.append((iiee[0].sum() + iiee[1].sum()) / climatological_ice_edge[concentration].loc[date])

        # iiee = IIEE(sic_forecast, sic_target, lsmask)
        # a_plus = iiee[0]
        # a_minus = iiee[1]

        area_dist_target = contourAreaDistribution(sic_target, lsmask, num_classes = config['num_outputs'], side_length = 1)
        area_dist_forecast = contourAreaDistribution(sic_forecast, lsmask, num_classes = config['num_outputs'], side_length = 1)

        # a_plus_minimum_distance = minimumDistanceToIceEdge(a_plus, ice_edge_target, lat, lon)
        # a_minus_minimum_distance = minimumDistanceToIceEdge(a_minus, ice_edge_target, lat, lon)

        output_list.append([pd.to_datetime(date, format="%Y%m%d"), target_length, forecast_length, *NIIEE, *area_dist_target.tolist(), *area_dist_forecast.tolist()])


    # output_df = pd.DataFrame(output_list, columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus', 'mean_minimum_distance_to_ice_edge_a_plus', 'mean_minimum_distance_to_ice_edge_a_minus', 'target_area0', 'target_area1', 'target_area2', 'target_area3', 'target_area4', 'target_area5', 'target_area6', 'forecast_area0', 'forecast_area1', 'forecast_area2', 'forecast_area3', 'forecast_area4', 'forecast_area5', 'forecast_area6'])
    output_df = pd.DataFrame(output_list, columns = ['date', 'target_length', 'forecast_length', *[f"NIIEE_{i}" for i in range(1, config['num_outputs'])], *[f"target_area{i}" for i in range(config['num_outputs'])], *[f"forecast_area{i}" for i in range(config['num_outputs'])]])

    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}{model_name}.csv")
    


if __name__ == "__main__":
    main()
