import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET')

import os
import pandas as pd
import glob
import h5py

import numpy as np

from calendar import monthrange
from verification_metrics import IIEE_fast, find_ice_edge, ice_edge_length, contourAreaDistribution, minimumDistanceToIceEdge
from tqdm import tqdm
from helper_functions import read_config_from_csv
from datetime import datetime, timedelta
from netCDF4 import Dataset

def load_barents(yyyymmdd, lead_time):
    # Currently only returns arome member
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/barents/"
    PATH_TARGET = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/targets/"

    yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
    yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = lead_time - 1)).strftime('%Y%m%d')
    yyyymmdd_ml = (yyyymmdd_datetime - timedelta(days = 1)).strftime('%Y%m%d')

    barents_path = glob.glob(f"{PATH_FORECAST}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/barents_mean_b{yyyymmdd}.nc")[0]

    target_path = glob.glob(f"{PATH_TARGET}{yyyymmdd_valid[:4]}/{yyyymmdd_valid[4:6]}/target_v{yyyymmdd_valid}.nc")[0]

    with Dataset(barents_path, 'r') as nc:
        barents_sic = nc.variables['sic'][lead_time - 1, 0, :,:]

    with Dataset(target_path, 'r') as nc:
        target_sic = nc.variables['sic'][:,:]

    return barents_sic, target_sic, yyyymmdd_ml


def load_ml(yyyymmdd, lead_time):
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/ml/lead_time_2/osisaf_trend_5/weights_05011118/"
    PATH_TARGET = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/targets/"

    yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
    yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = lead_time)).strftime('%Y%m%d')

    ml_path = glob.glob(f"{PATH_FORECAST}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/{yyyymmdd_valid}_b{yyyymmdd}.nc")[0]

    target_path = glob.glob(f"{PATH_TARGET}{yyyymmdd_valid[:4]}/{yyyymmdd_valid[4:6]}/target_v{yyyymmdd_valid}.nc")[0]

    with Dataset(ml_path, 'r') as nc:
        ml_sic = nc.variables['sic'][:,:]

    with Dataset(target_path, 'r') as nc:
        target_sic = nc.variables['sic'][:,:]

    return ml_sic, target_sic, yyyymmdd

def load_nextsim(yyyymmdd, lead_time):
    # Currently only returns arome member
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/"
    PATH_TARGET = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/targets/"

    yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
    yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = lead_time - 1)).strftime('%Y%m%d')
    yyyymmdd_ml = (yyyymmdd_datetime - timedelta(days = 1)).strftime('%Y%m%d')

    nextsim_path = glob.glob(f"{PATH_FORECAST}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/nextsim_mean_b{yyyymmdd}.nc")[0]

    target_path = glob.glob(f"{PATH_TARGET}{yyyymmdd_valid[:4]}/{yyyymmdd_valid[4:6]}/target_v{yyyymmdd_valid}.nc")[0]

    with Dataset(nextsim_path, 'r') as nc:
        nextsim_sic = nc.variables['sic'][lead_time - 1, :,:]

    with Dataset(target_path, 'r') as nc:
        target_sic = nc.variables['sic'][:,:]

    return nextsim_sic, target_sic, yyyymmdd_ml

def load_osisaf(yyyymmdd, lead_time):
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/osisaf/osisaf_trend_5/"
    PATH_TARGET = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/targets/"

    yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
    yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = lead_time)).strftime('%Y%m%d')

    osisaf_path = glob.glob(f"{PATH_FORECAST}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/osisaf_mean_b{yyyymmdd}.nc")[0]

    target_path = glob.glob(f"{PATH_TARGET}{yyyymmdd_valid[:4]}/{yyyymmdd_valid[4:6]}/target_v{yyyymmdd_valid}.nc")[0]

    with Dataset(osisaf_path, 'r') as nc:
        osisaf_sic = nc.variables['sic'][lead_time - 1, :,:]

    with Dataset(target_path, 'r') as nc:
        target_sic = nc.variables['sic'][:,:]

    return osisaf_sic, target_sic, yyyymmdd

def load_persistence(yyyymmdd, lead_time):
    PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/targets/"
    PATH_TARGET = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/targets/"

    yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
    yyyymmdd_valid = (yyyymmdd_datetime + timedelta(days = lead_time)).strftime('%Y%m%d')

    persistence_path = glob.glob(f"{PATH_FORECAST}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/target_v{yyyymmdd}.nc")[0]

    target_path = glob.glob(f"{PATH_TARGET}{yyyymmdd_valid[:4]}/{yyyymmdd_valid[4:6]}/target_v{yyyymmdd_valid}.nc")[0]

    with Dataset(persistence_path, 'r') as nc:
        persistence_sic = nc.variables['sic'][:,:]

    with Dataset(target_path, 'r') as nc:
        target_sic = nc.variables['sic'][:,:]

    return persistence_sic, target_sic, yyyymmdd


def main():
    product = sys.argv[1]
    lead_time = int(sys.argv[2])

    load_func = None
    
    osisaf_trend = None

    if product == 'barents':
        load_func = load_barents

    elif product == 'ml':
        load_func = load_ml
        osisaf_trend = 5
        # assert len(sys.argv) > 3, "To compute ml stats, supply valid model string"
        # path_config = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/"
        # weights = "weights_05011118"
        # config = read_config_from_csv(f"{path_config}configs/{weights}.csv")
        # osisaf_trend = config['osisaf_trend']

    elif product == 'nextsim':
        # PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/"
        load_func = load_nextsim

    elif product == 'osisaf':
        # assert len(sys.argv) > 3, "To compute osisaf stats, supply osisaf trend length"
        osisaf_trend = 5
        # PATH_FORECAST = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/osisaf/osisaf_trend_{osisaf_trend}/"
        load_func = load_osisaf

    elif product == 'persistence':
        load_func = load_persistence

    else:
        print("No valid product supplied")
        exit()


    PATH_OUTPUTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/lead_time_{lead_time}/"

    PATH_COMMONS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/commons.nc"

    if product == 'ml' or product == 'osisaf':
        PATH_OUTPUTS += f"osisaf_trend_{osisaf_trend}/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    output_list = []

    with Dataset(PATH_COMMONS, 'r') as nc:
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        lsmask = nc.variables['lsmask'][:,:]

    year = 2022
    months = []
    days = []
    for month in range(1, 13):
        months.append(month)
        days.append(monthrange(int(year), (month))[1])
 
    for i, month in enumerate(months):
        for dd in range(1, days[i] + 1):
        # for dd in range(1, 4):
            # Load time of target
            # Assume forecast initiated lead_time - 1 day before target valid
            yyyymmdd = f"{year}{month:02d}{dd:02d}"
            print(yyyymmdd)

            try:
                sic_forecast, sic_target, yyyymmdd_ml = load_func(yyyymmdd, lead_time)

            except IndexError:
                continue

            # ice_edge_target = find_ice_edge(sic_target, lsmask)
            # target_length = ice_edge_length(ice_edge_target, s = 3)

            # ice_edge_forecast = find_ice_edge(sic_forecast, lsmask)
            # forecast_length = ice_edge_length(ice_edge_forecast, s = 3)
            IIEE = []
            for i in range(1, 7):
                iiee = IIEE_fast(sic_forecast, sic_target, lsmask, side_length = 3, threshold = i)
                IIEE.append(iiee[0].sum() + iiee[1].sum())

            # area_dist_target = contourAreaDistribution(sic_target, lsmask, side_length = 3)
            # area_dist_forecast = contourAreaDistribution(sic_forecast, lsmask, side_length = 3)

            # a_plus_minimum_distance = minimumDistanceToIceEdge(a_plus, ice_edge_target, lat, lon)
            # a_minus_minimum_distance = minimumDistanceToIceEdge(a_minus, ice_edge_target, lat, lon)

            # output_list.append([pd.to_datetime(yyyymmdd_ml, format="%Y%m%d"), target_length, forecast_length, np.mean([target_length, forecast_length]), a_plus.sum() + a_minus.sum(), a_plus.sum(), a_minus.sum(), a_plus_minimum_distance.mean(), a_minus_minimum_distance.mean()] + area_dist_target.tolist() + area_dist_forecast.tolist())
            output_list.append([pd.to_datetime(yyyymmdd_ml, format="%Y%m%d"), *IIEE])


    # output_df = pd.DataFrame(output_list, columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus', 'mean_minimum_distance_to_ice_edge_a_plus', 'mean_minimum_distance_to_ice_edge_a_minus', 'target_area0', 'target_area1', 'target_area2', 'target_area3', 'target_area4', 'target_area5', 'target_area6', 'forecast_area0', 'forecast_area1', 'forecast_area2', 'forecast_area3', 'forecast_area4', 'forecast_area5', 'forecast_area6'])
    output_df = pd.DataFrame(output_list, columns = ['date', 'IIEE_1', 'IIEE_2', 'IIEE_3', 'IIEE_4', 'IIEE_5', 'IIEE_6'])

    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}{product}.csv")
    


if __name__ == "__main__":
    main()