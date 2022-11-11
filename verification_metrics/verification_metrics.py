import os
import glob
from re import A
import h5py

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def find_ice_edge(sic, mask, threshold = 2):
    """Calculates and defines a ice_edge mask for a given array

    Args:
        sic (array): input sic field
        threshold : sic class. Defaults to 2, i.e. very open ice

    Returns:
        array: binary sea ice edge array
    """

    mask_padded = np.pad(mask, 1, 'constant', constant_values = 1)
    sic_padded = np.pad(sic, 1, 'constant', constant_values = 7)
    sic_padded = np.where(mask_padded == 1, 7, sic_padded)

    ice_edge = np.zeros_like(sic_padded[1:-1, 1:-1])
    H, W = sic_padded.shape
    plt.contourf(sic_padded, cmap='cividis')
    for i in range(1, H-1):
        for j in range(1, W-1):
            current = sic_padded[i,j]
            if current == 7:
                continue

            left = sic_padded[i, j-1]
            right = sic_padded[i, j+1]
            top = sic_padded[i-1, j]
            bottom = sic_padded[i+1, j]

            neighbor = np.array([top, bottom, left, right])

            smallest_neighbor = np.min(neighbor)

            # print(f"pixel [{i-1}, {j-1}], smallest neighbor = {smallest_neighbor}", end='\r')

            ice_edge[i-1,j-1] = ((current >= threshold) and (smallest_neighbor < threshold)).astype(int)

    # print('\n')
    ice_edge[:200, 1500:] = 0
    return ice_edge

def find_ice_edge_from_fraction(sic, mask, threshold = 15):
    """Calculates and defines a ice_edge mask for a given array of fractional sic

    Args:
        sic (array): input sic field
        threshold : sic fraction. Defaults to 0.15

    Returns:
        array: binary sea ice edge array
    """

    mask_padded = np.pad(mask, 1, 'constant', constant_values = 1)
    sic_padded = np.pad(sic, 1, 'constant', constant_values = np.nan)
    sic_padded[mask_padded] = np.nan

    ice_edge = np.zeros_like(sic_padded[1:-1, 1:-1])
    H, W = sic_padded.shape

    for i in range(1, H-1):
        for j in range(1, W-1):
            current = sic_padded[i,j]
            if current == np.nan:
                continue

            left = sic_padded[i, j-1]
            right = sic_padded[i, j+1]
            top = sic_padded[i-1, j]
            bottom = sic_padded[i+1, j]

            neighbor = np.array([top, bottom, left, right])

            smallest_neighbor = np.min(neighbor)

            # print(f"pixel [{i-1}, {j-1}], smallest neighbor = {smallest_neighbor}", end='\r')

            ice_edge[i-1,j-1] = ((current >= threshold) and (smallest_neighbor < threshold)).astype(int)

    # print('\n')
    # ice_edge[:200, 1500:] = 0
    return ice_edge

def calculate_distance(current_ice_edge, other_ice_edge, x, y):
    d = {'distance': [], 'i': [], 'j': []}
    I_current, J_current = np.where(current_ice_edge == 1)
    I_other, J_other = np.where(other_ice_edge == 1)

    for idx, i_current, j_current in zip(range(0, len(I_current)), I_current,J_current):
        print(f"cell {idx} / {len(I_current)}", end = '\r')
        d['i'].append(i_current)
        d['j'].append(j_current)

        distances = np.zeros_like(I_other)

        x_term = np.power(x[J_other] - x[j_current], 2)
        y_term = np.power(y[I_other] - y[i_current], 2)
        distances = np.sqrt(x_term + y_term)

        d['distance'].append(distances.min())

    return d


def IIEE(sic_prediction, sic_target, mask, a = 1, threshold = 2):
    mask[:200, 1500:] = 1     # Baltic mask
    sic_target_masked = np.ma.masked_array(sic_target, mask=mask)
    sic_prediction_masked = np.ma.masked_array(sic_prediction, mask=mask)

    a_plus = np.logical_and(np.greater_equal(sic_prediction_masked, threshold), np.less(sic_target_masked, threshold)).astype(int)
    a_minus = np.logical_and(np.greater_equal(sic_target_masked, threshold), np.less(sic_prediction_masked, threshold)).astype(int)
    ocean = np.logical_and(np.less(sic_prediction_masked, threshold), np.less(sic_target_masked, threshold)).astype(int)
    ice = np.logical_and(np.greater_equal(sic_prediction_masked, threshold), np.greater_equal(sic_target_masked, threshold)).astype(int)

    return np.ma.stack((a_plus*(a**2), a_minus*(a**2), ocean, ice))

def ice_edge_length(ice_edge, s = 1):
    ice_edge = np.pad(ice_edge, 1, 'constant')
    I, J = np.where(ice_edge == 1)
    length = 0.
    sqrt_two = np.sqrt(2)
    
    for i,j in np.nditer((I,J)):
        top = ice_edge[i-1, j]
        bottom = ice_edge[i+1, j]
        left = ice_edge[i, j-1]
        right = ice_edge[i, j+1]

        neighbors = top + bottom + left + right

        if neighbors == 0:
            length += sqrt_two * s

        elif neighbors == 1:
            length += 0.5*(s + sqrt_two)

        else:
            length += s

    return 1000. * length

def contourAreaDistribution(icefield, mask, num_classes=7, side_length = 1):
    contour_matrix = np.zeros((*icefield.shape, num_classes))
    icefield = np.where(mask == 1, 7, icefield)

    for i in range(num_classes):
        contour_matrix[...,i] = np.where(icefield == i, 1, 0)

    area_array = (side_length**2) * np.sum(contour_matrix.reshape((np.prod(icefield.shape), num_classes)), axis=0)

    return area_array

def greatCircleDistance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes the Great Circle Distance between two points 
        AUTHOR: Cyril Palerme

    Args:
        lon1 (float): longitude point1
        lat1 (float): latitude point1
        lon2 (float): longitude point2
        lat2 (float): latitude point2

    Returns:
        float: Great Circle Distance between point1 and point2
    """

    # Convert from degrees to radians
    pi = 3.14159265
    lon1 = lon1 * 2 * pi / 360.
    lat1 = lat1 * 2 * pi / 360.
    lon2 = lon2 * 2 * pi / 360.
    lat2 = lat2 * 2 * pi / 360.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6.371e6 * c
    return distance



def test_DistanceToIceEdge():
    # Define global paths
    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_PREDICTIONS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/weights_05111353/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/test_data/"

    year = "2021"
    month = "**"

    southern_boundary = 578
    eastern_boundary = 1792

    target_path = sorted(glob.glob(f"{PATH_TARGETS}{year}/{month}/*.hdf5"))[0]
    prediction_path = sorted(glob.glob(f"{PATH_PREDICTIONS}{year}/{month}/*.hdf5"))[0]

    with h5py.File(target_path, 'r') as infile_target:
        sic_target = infile_target['sic_target'][southern_boundary:, :eastern_boundary]
        lsmask = infile_target['lsmask'][southern_boundary:, :eastern_boundary]
        lat = infile_target['lat'][southern_boundary:, :eastern_boundary]
        lon = infile_target['lon'][southern_boundary:, :eastern_boundary]

    with h5py.File(prediction_path, 'r') as infile_forecast:
        sic_forecast = infile_forecast['y_pred'][0]

    target_ice_edge = find_ice_edge(sic_target, lsmask)
    plt.contourf(target_ice_edge)
    plt.show()
    print(ice_edge_length(target_ice_edge))


def main():
    
    # Define global paths
    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/one_day_forecast/"
    PATH_PREDICTIONS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/Data/weights_20091742/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/"

    year = "2021"
    month = "01"

    target_path = sorted(glob.glob(f"{PATH_TARGETS}{year}/{month}/*.hdf5"))[1]
    prediction_path = sorted(glob.glob(f"{PATH_PREDICTIONS}{year}/{month}/*.hdf5"))[0]

    with h5py.File(target_path, 'r') as infile_target:
        sic_target = infile_target['sic_target'][451::2, :1792:2]
        lsmask = infile_target['lsmask'][451::2, :1792:2]
        x = infile_target['x'][:1792:2]
        y = infile_target['y'][451::2]


    with h5py.File(prediction_path, 'r') as infile_pred:
        sic_pred = infile_pred['y_pred'][0]

    if not os.path.exists("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/2021/01/ice_edge_20210105.hdf5"):
        ice_edge_target = find_ice_edge(sic_target, lsmask)
        ice_edge_pred = find_ice_edge(sic_pred, lsmask)


        if not os.path.exists(f"{PATH_OUTPUTS}{year}/{month}/"):
            os.makedirs(f"{PATH_OUTPUTS}{year}/{month}/")

        with h5py.File(f"{PATH_OUTPUTS}{year}/{month}/ice_edge_{target_path[-13:-5]}.hdf5", "w") as outfile:
            outfile['ice_edge_target'] = ice_edge_target
            outfile['ice_edge_pred'] = ice_edge_pred
    
    else:
        with h5py.File("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/2021/01/ice_edge_20210105.hdf5") as f:
            ice_edge_target = f['ice_edge_target'][:]
            ice_edge_pred = f['ice_edge_pred'][:]


    d_pred = calculate_distance(ice_edge_pred, ice_edge_target, x, y)
    d_target = calculate_distance(ice_edge_target, ice_edge_pred, x, y)
    
    
    print("\n")
    print("D_RMS^IE")
    print(0.5*(np.sqrt(np.sum(np.power(d_pred['distance'],2))/len(d_pred['distance'])) + np.sqrt(np.sum(np.power(d_target['distance'],2))/len(d_target['distance']))))
    print("D_AVG^IE")
    print(0.5*(np.mean(d_pred['distance']) + np.mean(d_target['distance'])))
    
    length_pred = ice_edge_length(ice_edge_pred)
    length_target = ice_edge_length(ice_edge_target)

    print(f"{length_pred=}")
    print(f"{length_target=}")

    iiee = IIEE(sic_pred, sic_target, lsmask)

    A_plus = iiee[0].sum()
    A_minus = iiee[1].sum()

    A_iiee = A_plus + A_minus
    alpha_iiee = A_plus - A_minus

    D_iiee = (2 * A_iiee) / (length_target + length_pred)
    delta = (2 * alpha_iiee) / (length_target + length_pred)

    print(f"{D_iiee=}")
    print(f"{delta=}")





if __name__ == "__main__":
    test_DistanceToIceEdge()
    # main()