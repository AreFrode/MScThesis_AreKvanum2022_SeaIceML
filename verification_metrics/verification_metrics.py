import os
import glob
from re import A
import h5py

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def find_ice_edge(sic, mask, threshold: int = 2):
    """Creates an Ice-Edge mask containing spatially aware ice-edge pixels,
        code inspired by derivation performed in [Melsom, 2019]

    Args:
        sic (array): input sic field
        mask (array): arbitrary mask (usually land sea mask)
        threshold: sic class thresholding the ice edge. Defaults to 2, i.e. very open ice

    Returns:
        array: sea ice edge mask
    """

    mask_padded = np.pad(mask, 1, 'constant', constant_values = 1)
    sic_padded = np.pad(sic, 1, 'constant', constant_values = 7)
    sic_padded = np.where(mask_padded == 1, 7, sic_padded)

    ice_edge = np.zeros_like(sic_padded[1:-1, 1:-1])
    H, W = sic_padded.shape
    # plt.figure()
    # cbar = plt.pcolormesh(sic_padded == 7)
    # plt.colorbar(cbar)
    # plt.savefig('ice_edge.png')
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
    # ice_edge[:200, 1500:] = 0
    return ice_edge

def find_ice_edge_from_fraction(sic, mask, threshold = 15):
    """Calculates and defines a ice_edge mask for a given array containing fractional sic values

    Args:
        sic (array): input fractional sic field
        mask (array): arbitrary mask (usually land sea mask)
        threshold : sic fraction for ice edge thresholding. Defaults to 0.15

    Returns:
        array: sea ice edge mask
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

    return ice_edge

def calculate_distance(current_ice_edge, other_ice_edge, x, y):
    """computes the euclidean distance to the nearest ice edge grid cell between two products for all ice edge cells (only correct for equidistant grid)
    Code implemnted according to derivation in [Melsom, 2019]

    Args:
        current_ice_edge (array): Ice edge grid cells to loop through
        other_ice_edge (array): Ice edge grid cells to compute distance against
        x (array): x-values
        y (array): y-values

    Returns:
        array: distance to the nearest "other" ice edge cell for all "current" ice edge cells
    """

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


def IIEE(sic_prediction, sic_target, mask, a: int = 1, threshold: int = 2):
    """Computes the IIEE between two SIC fields, implemented according to [Goessling, 2016]

    Args:
        sic_prediction (array): SIC field serving as the forecast
        sic_target (array): SIC field serving as the target
        mask (lsmask): Arbitrary mask (usually land sea mask)
        a (int, optional): Grid cell side length in km. Defaults to 1.
        threshold (int, optional): SIC Class thresholding open water and sea ice. Defaults to 2.

    Returns:
        Masked array: Stack consisting of [a_plus, a_minus, ocean, ice], cell values in km
    """

    # mask[:200, 1500:] = 1     # Baltic mask
    sic_target_masked = np.ma.masked_array(sic_target, mask=mask)
    sic_prediction_masked = np.ma.masked_array(sic_prediction, mask=mask)

    a_plus = np.logical_and(np.greater_equal(sic_prediction_masked, threshold), np.less(sic_target_masked, threshold)).astype(int)
    a_minus = np.logical_and(np.greater_equal(sic_target_masked, threshold), np.less(sic_prediction_masked, threshold)).astype(int)
    ocean = np.logical_and(np.less(sic_prediction_masked, threshold), np.less(sic_target_masked, threshold)).astype(int)
    ice = np.logical_and(np.greater_equal(sic_prediction_masked, threshold), np.greater_equal(sic_target_masked, threshold)).astype(int)

    return np.ma.stack((a_plus*(a**2), a_minus*(a**2), ocean, ice))

def ice_edge_length(ice_edge, s = 1) -> float:
    """Computes the ice edge length given a ice_edge mask array, 
    implemented according to [Melsom, 2019]

    Args:
        ice_edge (array): Ice Edge mask array
        s (int, optional): Grid cell side length in km. Defaults to 1.

    Returns:
        float: ice edge length in meter
    """

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
    """Computes the area for each sea ice concentration class contour

    Args:
        icefield (array): Sea Ice Concentration field
        mask (array): Arbitrary mask (usually land sea mask)
        num_classes (int, optional): Number of sea ice classes. Defaults to 7.
        side_length (int, optional): Grid cell side length in km. Defaults to 1.

    Returns:
        array: 1d-array containing contour area for each respective index
    """

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
    distance = 6.371e6 * c   # Convert to meter
    return distance


def minimumDistanceToIceEdge(wrongly_classified_field, target_ice_edge, lat, lon):
    """Computes the great circle distance between a set of pixels and their nearest ice edge pixel

    Args:
        wrongly_classified_field (array): binary array (usually a_plus / a_minus)
        target_ice_edge (array): Ice edge set for the ground truth 
        lat (array): Region latitude array
        lon (array): Regional longitude array

    Returns:
        array: 1-d array containing minimum distance to target ice edge for each pixel in wrongly classified field
    """

    wrongly_classified_field = np.ma.filled(wrongly_classified_field, 0)

    wrongly_classified_indexes = np.where(wrongly_classified_field != 0)
    ice_edge_indexes = np.where(target_ice_edge == 1)
    ice_edge_indexes_stacked = np.dstack((ice_edge_indexes[0], ice_edge_indexes[1]))[0]

    min_distances = []

    for idx in np.dstack((wrongly_classified_indexes[0], wrongly_classified_indexes[1]))[0]:
        idx_full = np.full_like(ice_edge_indexes_stacked, idx).T
        idx_tuple = (idx_full[0], idx_full[1])

        min_distances.append(greatCircleDistance(
            lon[idx_tuple], 
            lat[idx_tuple],
            lon[ice_edge_indexes],
            lat[ice_edge_indexes]
        ).min())

    return np.array(min_distances)


def coarse_grid_cell_ice_edge_fraction(neighborhood, n):
    H,W = neighborhood.shape

    for i in range(0, H, n):
        for j in range(0, W, n):
            view = neighborhood[i:i+3,j:j+3]
            print(view)


def test_DistanceToIceEdge():
    observed = np.array([[0,0,1,0,0,0], 
                         [0,0,1,0,0,0],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,1],
                         [0,0,0,0,0,0],
                         [0,0,0,0,0,0]])

    modeled = np.array([[0,0,0,0,0,0],
                         [1,1,0,0,0,0],
                         [0,0,1,1,0,0],
                         [0,0,0,0,1,1],
                         [0,0,0,0,0,1],
                         [0,0,0,0,0,0]])

    view = coarse_grid_cell_ice_edge_fraction(observed, n = 3)
    print(view)

    exit()

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

    forecast_ice_edge = find_ice_edge(sic_forecast, lsmask)
    target_ice_edge = find_ice_edge(sic_target, lsmask)


    # iiee = IIEE(sic_forecast, sic_target, lsmask)
    # a_plus = iiee[0]
    # a_minus = iiee[1]

    # print(a_plus.sum())
    # print(a_minus.sum())

    # plt.pcolormesh(a_plus)
    # plt.savefig('a_plus.png')

    # plt.pcolormesh(a_minus)
    # plt.savefig('a_minus.png')

    # a_plus_minimum_distance = minimumDistanceToIceEdge(a_plus, target_ice_edge, lat, lon)
    # print(a_plus_minimum_distance.mean())

    # a_minus_minumum_distance = minimumDistanceToIceEdge(a_minus, target_ice_edge, lat, lon)
    # print(a_minus_minumum_distance.mean())


    
def main():
    
    # Define global paths
    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/"
    PATH_PREDICTIONS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/outputs/Data/weights_28112014/"
    PATH_OUTPUTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/test_data/"

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

    


    """
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
    """




if __name__ == "__main__":
    test_DistanceToIceEdge()
    # main()