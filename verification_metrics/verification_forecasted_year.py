import os
import glob
import h5py

import numpy as np
import pandas as pd
import verification_metrics as ice_edge_metrics

from datetime import datetime, timedelta


def main():
    # Define global paths
    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/one_day_forecast/"
    PATH_PREDICTIONS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/Data/weights_20091742/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/"

    year = "2021"
    month = "01"

    # Setup data streams
    prediction_path = sorted(glob.glob(f"{PATH_PREDICTIONS}{year}/{month}/*.hdf5"))[:10]
    
    output_dataframe = pd.DataFrame(columns=['D_RMS^IE', 'D_AVG^IE', 'ice_edge_length_prediction', 'ice_edge_length_target', 'D_iiee', 'delta_iiee'])

    # Prepare constant fields

    with h5py.File(f"{PATH_TARGETS}2021/01/PreparedSample_20210104.hdf5", 'r') as f:
        x = f['x'][:1792:2]
        y = f['y'][451::2]
        lsmask = f['lsmask'][451::2, :1792:2]
        
    
    # Compute statistics for all samples

    for path in prediction_path:
        yyyymmdd = path[-17:-9]
        print(f"{yyyymmdd}", end='\r')

        print(f"{yyyymmdd}")
        yyyymmdd_datetime = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd_previous = (yyyymmdd_datetime - timedelta(days = 1)).strftime('%Y%m%d')

        with h5py.File(path, 'r') as f_prediction:
            sic_prediction = f_prediction['y_pred'][0]

        with h5py.File(f"{PATH_TARGETS}{yyyymmdd_previous[:4]}/{yyyymmdd_previous[4:6]}/PreparedSample_{yyyymmdd_previous}.hdf5", 'r') as f_target:
            sic_target = f_target['sic_target'][451::2, :1792:2]

        # Ice Edge Displacement Metrics

        ice_edge_target = ice_edge_metrics.find_ice_edge(sic_target, lsmask)
        ice_edge_prediction = ice_edge_metrics.find_ice_edge(sic_prediction, lsmask)

        if not os.path.exists(f"{PATH_OUTPUTS}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"):
            os.makedirs(f"{PATH_OUTPUTS}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/")

        with h5py.File(f"{PATH_OUTPUTS}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/ice_edge_{yyyymmdd}.hdf5", "w") as outfile_ice_edge:
            outfile_ice_edge['ice_edge_target'] = ice_edge_target
            outfile_ice_edge['ice_edge_pred'] = ice_edge_prediction

        d_pred = ice_edge_metrics.calculate_distance(ice_edge_prediction, ice_edge_target, x, y)
        d_target = ice_edge_metrics.calculate_distance(ice_edge_target, ice_edge_prediction, x, y)

        D_rms = 0.5*(np.sqrt(np.sum(np.power(d_pred['distance'],2))/len(d_pred['distance'])) + np.sqrt(np.sum(np.power(d_target['distance'],2))/len(d_target['distance'])))

        D_avg = 0.5*(np.mean(d_pred['distance']) + np.mean(d_target['distance']))

        length_prediction = ice_edge_metrics.ice_edge_length(ice_edge_prediction)
        length_target = ice_edge_metrics.ice_edge_length(ice_edge_target)

        # IIEE Metrics

        iiee = ice_edge_metrics.IIEE(sic_prediction, sic_target, lsmask)

        if not os.path.exists(f"{PATH_OUTPUTS}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"):
            os.makedirs(f"{PATH_OUTPUTS}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/")

        with h5py.File(f"{PATH_OUTPUTS}{yyyymmdd[:4]}/{yyyymmdd[4:6]}/iiee_{yyyymmdd}.hdf5", 'w') as outfile_iiee:
            # HDF5 can not handle masked arrays, replace mask with invalid integer (-1)
        
            iiee = iiee.filled(-1)
            outfile_iiee['a_plus'] = iiee[0]
            outfile_iiee['a_minus'] = iiee[1]
            outfile_iiee['ocean'] = iiee[2]
            outfile_iiee['ice'] = iiee[3]


        A_plus = iiee[0].sum()
        A_minus = iiee[1].sum()

        A_iiee = A_plus + A_minus
        alpha_iiee = A_plus - A_minus

        D_iiee = (2 * A_iiee) / (length_target + length_prediction)
        delta = (2 * alpha_iiee) / (length_target + length_prediction)

        output_dictionary = {'D_RMS^IE': D_rms, 'D_AVG^IE': D_avg, 'ice_edge_length_prediction': length_prediction, 'ice_edge_length_target': length_target, 'D_iiee': D_iiee, 'delta_iiee': delta}

        output_dataframe.loc[len(output_dataframe.index)] = output_dictionary

    if not os.path.exists(f"{PATH_OUTPUTS}verification_tables/"):
        os.makedirs(f"{PATH_OUTPUTS}verification_tables/")

    output_dataframe.to_csv(f"{PATH_OUTPUTS}/verification_tables/first_attempt_202101.csv")




if __name__ == "__main__":
    main()