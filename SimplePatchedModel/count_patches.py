import glob
import  h5py
import numpy as np

def runstuff():
    path_data = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ProjToPatch/Data/"

    patches = {'below': 0, 'between': 0, 'above': 0}
    for file in glob.glob(f"{path_data}*.hdf5"):
        print(f"{file=}")
        f = h5py.File(file, 'r')

        for thresh in f.keys():
            print(f"{thresh=}")
            tmp = 0
            for yyyymmdd in f[f"{thresh}"].keys():
                data = f[f"{thresh}/{yyyymmdd}"]
                xc = data["xc"]
                tmp += xc.shape[0]
     

            print(f"{tmp=}")


        
        f.close()
        break


if __name__ == "__main__":
    runstuff()