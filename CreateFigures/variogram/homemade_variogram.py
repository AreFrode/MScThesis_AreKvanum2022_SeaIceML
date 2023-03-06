import os

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from netCDF4 import Dataset


def compute_spatial_correlation(field, filter_size):
    sizex, sizey = field.shape
    filter_edge = int(np.floor(0.5 * filter_size))
    points_x = np.random.randint(filter_edge, sizex - filter_edge, size = 1000)
    points_y = np.random.randint(filter_edge, sizey - filter_edge, size = 1000)

    # print(f"{points_x.min()}, {points_x.max()}, {points_y.min()}, {points_y.max()}")

    # print(fields[0, points_y.max() + int(filter_edge), points_x.min() + int(filter_edge)])


    # print(f"{ points_x.max() - filter_edge}, {points_x.max() + filter_edge+1}, {points_y.max()-filter_edge}, {points_y.max()+filter_edge+1}")

    # neighborhood = fields[0, points_y.max()-filter_edge:points_y.max()+filter_edge+1, points_x.max() - filter_edge:points_x.max() + filter_edge+1]

    # print(neighborhood)

    # print(fields[0, 1791, 1791])

    # print(filter_edge % 2)

    if filter_size % 2 != 0:
        rhs = filter_edge+1
    else:
        rhs = filter_edge
    
    Mean = np.mean(field)
    Var = np.var(field)

    for coords in zip(points_y, points_x):

        neighborhood = field[coords[0]-filter_edge:coords[0]+rhs, coords[1] - filter_edge:coords[1] + rhs]

        center = neighborhood[filter_edge - 1, filter_edge - 1]
        
        neighborhood[filter_edge, filter_edge] = -999
        
        others = neighborhood[np.where(neighborhood != -999)]

        print(others)
    



        break
    

def main():
    # Define paths
    PATH_DATA = "/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/local_data/ICECHART_1kmAromeGrid_20220103T1500Z.nc"

    sample_rate = 1
    t = 0
    with Dataset(PATH_DATA, 'r') as nc:
        # x = nc.variables['x'][:1792:sample_rate]
        # y = nc.variables['y'][578::sample_rate]
        # t2m = nc.variables['t2m'][t,578::sample_rate,:1792:sample_rate]
        # xwind = nc.variables['xwind'][t,578::sample_rate,:1792:sample_rate]
        # ywind = nc.variables['ywind'][t,578::sample_rate,:1792:sample_rate]
        sic = nc.variables['sic'][578::sample_rate,:1792:sample_rate]


    compute_spatial_correlation(field = sic, filter_size = 6)

    exit()

if __name__ == "__main__":
    main()