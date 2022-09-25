import glob
import h5py
import os

import numpy as np

from unet import create_UNET
from dataset import HDF5Generator
from datetime import datetime, timedelta

def main():
    SEED_VALUE = 0
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/"
    PATH_DATA = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"

    BATCH_SIZE = 1
    
    data_2021 = np.array(sorted(glob.glob(f"{PATH_DATA}2021/**/*.hdf5", recursive=True)))

    validation_generator = HDF5Generator(data_2021, batch_size=BATCH_SIZE, shuffle=False)

    model = create_UNET(input_shape = (960, 896, 6), channels = [64, 128, 256, 512, 1024])

    load_status = model.load_weights(f"{PATH_OUTPUTS}models/weights_20091742").expect_partial()

    samples = len(validation_generator)

    for i in range(samples):
        print(f"Sample {i} of {samples}", end="\r")
        X, y = validation_generator[i]
        y_pred = np.argmax(model.predict(X), axis=-1)
        
        yyyymmdd = validation_generator.get_dates(i)[0][-13:-5]
        yyyymmdd = datetime.strptime(yyyymmdd, '%Y%m%d')
        yyyymmdd = (yyyymmdd + timedelta(days = 1)).strftime('%Y%m%d')

        hdf_path = f"{PATH_OUTPUTS}Data/{yyyymmdd[:4]}/{yyyymmdd[4:6]}/"
        if not os.path.exists(hdf_path):
            os.makedirs(hdf_path)

        output_file = h5py.File(f"{hdf_path}SIC_SimpleUNET_{yyyymmdd}T15Z.hdf5", "w-")

        output_file["y"] = y
        output_file["y_pred"] = y_pred
        output_file["date"] = yyyymmdd

        output_file.close()


if __name__ == "__main__":
    main()