import sys
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics')
sys.path.append('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset')

import os
import pandas as pd
import glob
import h5py

import numpy as np

from verification_metrics import IIEE_alt, find_ice_edge, ice_edge_length, contourAreaDistribution, root_mean_square_error
from tqdm import tqdm
from datetime import datetime, timedelta
from loadClimatologicalIceEdge import load_climatological_ice_edge

def main():
    lead_time = sys.argv[1]

    concentration = '15%'
    climatological_ice_edge = load_climatological_ice_edge(2022, concentration, int(lead_time))

    PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_{lead_time}/"
    PATH_OUTPUTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{lead_time}/"

    if not os.path.exists(PATH_OUTPUTS):
        os.makedirs(PATH_OUTPUTS)

    icecharts = sorted(glob.glob(f"{PATH_PERSISTENCE}2022/**/*.hdf5"))

    with h5py.File(icecharts[0], 'r') as constants:
        lsmask = constants['lsmask'][578:, :1792]

    output_list = []


    for icechart in tqdm(icecharts):
        yyyymmdd = icechart[-13:-5]

        with h5py.File(icechart, 'r') as infile:
            sic_target = infile['sic_target'][578:, :1792]
            sic_persistance = infile['sic'][578:, :1792]

        ice_edge_target = find_ice_edge(sic_target, lsmask)
        target_length = ice_edge_length(ice_edge_target)

        ice_edge_forecast = find_ice_edge(sic_persistance, lsmask)
        forecast_length = ice_edge_length(ice_edge_forecast)

        rmsd = root_mean_square_error(sic_persistance, sic_target, lsmask)

        NIIEE = []
        for i in range(1, 7):
            iiee = IIEE_alt(sic_persistance, sic_target, lsmask, side_length = 1, threshold = i)
            a_plus = iiee[0].sum()
            a_minus = iiee[1].sum()
            NIIEE.append((a_plus + a_minus) / climatological_ice_edge[concentration].loc[yyyymmdd])

        area_dist_target = contourAreaDistribution(sic_target, lsmask)
        area_dist_forecast = contourAreaDistribution(sic_persistance, lsmask)

        output_list.append([pd.to_datetime(yyyymmdd, format="%Y%m%d"), target_length, forecast_length, np.mean([target_length, forecast_length]), rmsd, *NIIEE, *area_dist_target.tolist(), *area_dist_forecast.tolist()])


    output_df = pd.DataFrame(output_list, columns = ['date', 'target_length', 'forecast_length', 'mean_length', 'rmse', *[f"NIIEE_{i}" for i in range(1, 7)], *[f"target_area{i}" for i in range(7)], *[f"forecast_area{i}" for i in range(7)]])

    output_df = output_df.set_index('date')
    output_df.to_csv(f"{PATH_OUTPUTS}persistence.csv")
    


if __name__ == "__main__":
    main()