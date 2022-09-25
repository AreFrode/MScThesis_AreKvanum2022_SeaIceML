import os
import glob
import h5py

import numpy as np

from verification_metrics import IIEE


def main():
    # Define global paths
    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"
    PATH_PREDICTIONS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/Data/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/"

    year = "2021"
    month = "01"

    target_path = sorted(glob.glob(f"{PATH_TARGETS}{year}/{month}/*.hdf5"))[1]
    prediction_path = sorted(glob.glob(f"{PATH_PREDICTIONS}{year}/{month}/*.hdf5"))[0]

    with h5py.File(target_path, 'r') as infile_target:
        sic_target = infile_target['sic_target'][451::2, :1792:2]
        lsmask = infile_target['lsmask'][451::2, :1792:2]


    with h5py.File(prediction_path, 'r') as infile_pred:
        sic_pred = infile_pred['y_pred'][0]

    sic_target = np.ma.masked_array(sic_target, mask=lsmask)
    sic_pred = np.ma.masked_array(sic_pred, mask=lsmask)

    iiee = IIEE(sic_pred, sic_target)
    print(f"{iiee[0].sum()=}")
    print(f"{iiee[1].sum()=}")




    if not os.path.exists(f"{PATH_OUTPUTS}{year}/{month}/"):
        os.makedirs(f"{PATH_OUTPUTS}{year}/{month}/")
    
    with h5py.File(f"{PATH_OUTPUTS}{year}/{month}/iiee_{target_path[-13:-5]}.hdf5", "w") as outfile:
        # HDF5 can not handle masked arrays, replace mask with invalid integer (-1)
        
        iiee = iiee.filled(-1)
        outfile['a_plus'] = iiee[0]
        outfile['a_minus'] = iiee[1]
        outfile['ocean'] = iiee[2]
        outfile['ice'] = iiee[3]


if __name__ == "__main__":
    main()