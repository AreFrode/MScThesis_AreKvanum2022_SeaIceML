import glob

import pandas as pd
import numpy as np

from netCDF4 import Dataset
from scipy.signal import correlate
from scipy.fft import fft2, ifft2, fftshift

def autocorrelate(Data):
    Data = np.array(Data)
    n = len(Data)
    Data_f = fft2(Data, s=((n*2) - 1, (n*2) - 1))

    for count in range(1, 10):
        i = np.arange(n - count)
        yield fftshift(np.multiply(Data_f[i], np.conjugate(Data_f[i+count]))).sum(axis=0).mean()


icecharts = sorted(glob.glob("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/RawIceChart_dataset/Data/2022/01/*.nc"))

Data = []
for icechart in icecharts:
    with Dataset(icechart, 'r') as nc:
        Data.append(nc.variables['sic'][::5,::5])

solution = list(autocorrelate(Data))
print(solution)
