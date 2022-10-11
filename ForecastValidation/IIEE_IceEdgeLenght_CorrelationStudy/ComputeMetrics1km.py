import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics")

import os
import glob
import h5py

import pandas as pd
import numpy as np

from verification_metrics import find_ice_edge, IIEE, ice_edge_length
from tqdm import tqdm

def main():
    # Define data-paths
    PATH_ICECHARTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"

    # Two day forecast with bottleneck at 1024 channels - architecture
    PATH_FORECASTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/weights_06101543/"

    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/IIEE_IceEdgeLenght_CorrelationStudy/Data/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    icecharts = sorted(glob.glob(f"{PATH_ICECHARTS}2021/**/*.hdf5", recursive = True))

    forecasts = sorted(glob.glob(f"{PATH_FORECASTS}2021/**/*.hdf5", recursive = True))

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask = constants['lsmask'][450:, :1840]


    output_df = pd.DataFrame(columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus'])
    for target, forecast in tqdm(zip(icecharts, forecasts), total=len(icecharts)):
        date = forecast[-17:-9]

        with h5py.File(target, 'r') as intarget:
            sic_target = intarget['sic_target'][450:, :1840]

        with h5py.File(forecast, 'r') as inforecast:
            sic_forecast = inforecast['y_pred'][0]

        ice_edge_target = find_ice_edge(sic_target, lsmask)
        target_length = ice_edge_length(ice_edge_target)

        ice_edge_forecast = find_ice_edge(sic_forecast, lsmask)
        forecast_length = ice_edge_length(ice_edge_forecast)

        iiee = IIEE(sic_forecast, sic_target, lsmask)
        a_plus = iiee[0].sum()
        a_minus = iiee[1].sum()

        tmp_df = pd.DataFrame([[pd.to_datetime(date, format="%Y%m%d"), target_length, forecast_length, np.mean([target_length, forecast_length]), a_plus + a_minus, a_plus, a_minus]], columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'IIEE', 'a_plus', 'a_minus'])

        output_df = pd.concat([output_df, tmp_df])
        

    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}1km_output.csv")

    
if __name__ == "__main__":
    main()