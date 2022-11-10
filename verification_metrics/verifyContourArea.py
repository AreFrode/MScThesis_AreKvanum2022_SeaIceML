import glob
import h5py

from verification_metrics import contourArea



def main():
    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/one_day_forecast/"

    year = "2021"
    month = "01"

    target_path = sorted(glob.glob(f"{PATH_TARGETS}{year}/{month}/*.hdf5"))[0]

    with h5py.File(target_path) as infile:
        target = infile['sic_target'][450:, :1840]

    out = contourArea(target)
    


if __name__ == "__main__":
    main()