import glob
import numpy as np
from netCDF4 import Dataset
from extract_invalid_region import get_invalid_coordinates

from tqdm import tqdm

from matplotlib import pyplot as plt


def main():
    paths = sorted(glob.glob(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ExternalIceConc_concentration_study/OSISAF/Data/**/*.nc", recursive=True))
    
    y_invalid, x_invalid = get_invalid_coordinates()

    # Compute mean over all months and store in (120, 1) array
    all_mean = np.zeros((120))

    for i, path in enumerate(tqdm((paths))):
        all_mean[i] = np.mean(Dataset(path, 'r').variables['ice_conc'][:][:, y_invalid, x_invalid])

    # Print total mean
    # print(f"Total mean: {np.mean(all_mean)}")
    # print(f"Total min: {np.min(all_mean)}")
    # print(f"Total max: {np.max(all_mean)}")

    # Print monthly mean
    # for i in range(a):
        # print(f"Montly mean month {i+1:02}: {np.mean(all_mean[i::12])}")
        # print(f"Montly min month {i+1:02}: {np.min(all_mean[i::12])}")
        # print(f"Montly max month {i+1:02}: {np.max(all_mean[i::12])}")

    fig, ax = plt.subplots(figsize=(15,15))
    for i in range(12):
        ax.scatter(np.repeat(i, 10), all_mean[i::12], c=range(10), cmap=plt.get_cmap('cividis'))

    ax.scatter()

    plt.savefig("Scatterplot.png")



if __name__ == "__main__":
    main()