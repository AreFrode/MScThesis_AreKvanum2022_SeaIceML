import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/two_day_forecast")

import os
import glob
import h5py

import pandas as pd
import numpy as np

from verification_metrics import find_ice_edge, IIEE, ice_edge_length
from tqdm import tqdm
from scipy.interpolate import griddata
from datetime import datetime, timedelta

def main():
    # Define data-paths
    PATH_ICECHARTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/IIEE_IceEdgeLenght_CorrelationStudy/Data/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    icecharts = sorted(glob.glob(f"{PATH_ICECHARTS}**/**/*.hdf5"))

    # Define 10km target grid
    x_min = 279103.2
    x_max = 2123103.2
    y_min = -897431.6
    y_max = 1471568.4

    nx = 1845
    ny = 2370

    x_input = np.linspace(x_min, x_max, nx)[:1792]
    y_input = np.linspace(y_min, y_max, ny)[578:]
    xx_input, yy_input = np.meshgrid(x_input, y_input)
    xx_input_flat = xx_input.flatten()
    yy_input_flat = yy_input.flatten()

    x_target = x_input[::10]
    y_target = y_input[::10]

    nx_target = len(x_target)
    ny_target = len(y_target)

    lsmask = np.zeros((ny_target, nx_target), dtype='int')

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask[...] = griddata((xx_input_flat, yy_input_flat), constants['lsmask'][578:, :1792].flatten(), (x_target[None, :], y_target[:, None]), method = 'nearest')

    output_df = pd.DataFrame(columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus'])
    for target in tqdm(icecharts):
        date = target[-13:-5]
        date = datetime.strptime(date, '%Y%m%d')
        date = (date + timedelta(days = 2)).strftime('%Y%m%d')
        sic_target = np.zeros((ny_target, nx_target))
        sic_forecast = np.zeros((ny_target, nx_target))

        with h5py.File(target, 'r') as intarget:
            sic_forecast[...] = griddata((xx_input_flat, yy_input_flat), intarget['sic'][578:, :1792].flatten(), (x_target[None, :], y_target[:, None]), method = 'nearest')
            sic_target[...] = griddata((xx_input_flat, yy_input_flat), intarget['sic_target'][578:, :1792].flatten(), (x_target[None, :], y_target[:, None]), method = 'nearest')

        ice_edge_target = find_ice_edge(sic_target, lsmask)
        target_length = ice_edge_length(ice_edge_target, s = 10)

        ice_edge_forecast = find_ice_edge(sic_forecast, lsmask)
        forecast_length = ice_edge_length(ice_edge_forecast, s = 10)

        iiee = IIEE(sic_forecast, sic_target, lsmask, a = 10)
        a_plus = iiee[0].sum()
        a_minus = iiee[1].sum()

        tmp_df = pd.DataFrame([[pd.to_datetime(date, format="%Y%m%d"), target_length, forecast_length, np.mean([target_length, forecast_length]), a_plus + a_minus, a_plus, a_minus]], columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus'])

        output_df = pd.concat([output_df, tmp_df])
        
    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}10km_persistance_output.csv")

    
if __name__ == "__main__":
    main()