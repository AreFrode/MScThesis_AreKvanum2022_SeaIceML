import h5py
import numpy as np

def runstuff():
    year = 2019
    month = 1
    yyyymmdd = 20190102
    f = h5py.File(f"Data/{year}/{month:02d}/PatchedAromeArcticBins_{yyyymmdd}.hdf5", 'r')
    # f = h5py.File(f"Data/backup/FullPatchedAromeArctic.hdf5", 'r')

    print(f['between']['lat'][0])


    f.close()


if __name__ == "__main__":
    runstuff()