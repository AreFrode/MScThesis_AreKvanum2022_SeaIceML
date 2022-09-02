import glob
import numpy as np
from netCDF4 import Dataset
from extract_invalid_region import get_invalid_coordinates

from tqdm import tqdm


def main():
    paths = sorted(glob.glob(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/OSISAF_concentration_study/Data/**/*.nc", recursive=True))
    
    y_invalid, x_invalid = get_invalid_coordinates()

    # Compute mean over all months and store in (120, 1) array
    all_mean = np.zeros((120))

    for i, path in enumerate(tqdm((paths))):
        all_mean[i] = np.mean(Dataset(path, 'r').variables['ice_conc'][:][:, y_invalid, x_invalid])

    # Print total mean
    print(f"Total mean: {np.mean(all_mean)}")

    # Print monthly mean
    for i in range(12):
        print(f"Montly mean month {i+1:02}: {np.mean(all_mean[i::12])}")


if __name__ == "__main__":
    main()