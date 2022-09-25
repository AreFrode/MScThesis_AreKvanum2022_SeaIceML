import os
import glob
import h5py

import numpy as np

def find_ice_edge(sic, threshold = 2):
    """Calculates and defines a ice_edge mask for a given array

    Args:
        sic (array): input sic field
        threshold : sic class. Defaults to 2, i.e. very open ice

    Returns:
        array: binary sea ice edge array
    """

    ice_edge = np.zeros_like(sic[1:-1, 1:-1])
    H, W = sic.shape

    for i in range(1, H-1):
        for j in range(1, W-1):
            current = sic[i,j]
            left = sic[i, j-1]
            right = sic[i, j+1]
            top = sic[i-1, j]
            bottom = sic[i+1, j]

            neighbor = np.array([top, bottom, left, right])

            smallest_neighbor = np.nanmin(neighbor)
            if np.isnan(smallest_neighbor):
                continue
            
            print(f"pixel [{i-1}, {j-1}], smallest neighbor = {smallest_neighbor}", end='\r')

            ice_edge[i-1,j-1] = ((current >= threshold) and (smallest_neighbor < threshold)).astype(int)

    print('\n')
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


def IIEE(sic_prediction, sic_target, threshold = 2):
    # I think this metric has a direction (not tested), the first is compared against the second
    a_plus = np.logical_and(np.greater_equal(sic_prediction, threshold), np.less(sic_target, threshold)).astype(int)
    a_minus = np.logical_and(np.greater_equal(sic_target, threshold), np.less(sic_prediction, threshold)).astype(int)
    ocean = np.logical_and(np.less(sic_prediction, threshold), np.less(sic_target, threshold)).astype(int)
    ice = np.logical_and(np.greater_equal(sic_prediction, threshold), np.greater_equal(sic_target, threshold)).astype(int)

    return np.ma.stack((a_plus, a_minus, ocean, ice))

def ice_edge_length(ice_edge, s = 1):
    # Here I assume that I use every other pixel, and their difference is 2 km
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
    


def main():
    
    # Define global paths
    PATH_TARGETS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"
    PATH_PREDICTIONS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/outputs/Data/"
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

    lsmask_padded = np.pad(lsmask, 1, 'constant', constant_values = 1)
    sic_target_padded = np.pad(sic_target, 1, 'constant', constant_values = 7)
    sic_target_padded = np.ma.masked_array(sic_target_padded, mask = lsmask_padded)

    sic_pred_padded = np.pad(sic_pred, 1, 'constant', constant_values = 7)
    sic_pred_padded = np.ma.masked_array(sic_pred_padded, mask = lsmask_padded)

    if not os.path.exists("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/2021/01/ice_edge_20210105.hdf5"):
        ice_edge_target = find_ice_edge(sic_target_padded)
        ice_edge_pred = find_ice_edge(sic_pred_padded)


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
    """
    print("D_RMS^IE")
    print(0.5*(np.sqrt(np.sum(np.power(d_pred['distance'],2))/len(d_pred['distance'])) + np.sqrt(np.sum(np.power(d_target['distance'],2))/len(d_target['distance']))))
    print("D_AVG^IE")
    print(0.5*(np.mean(d_pred['distance']) + np.mean(d_target['distance'])))
    """
    
    length_pred = ice_edge_length(ice_edge_pred)
    length_target = ice_edge_length(ice_edge_target)

    # print(f"{length_pred=}")
    # print(f"{length_target=}")

    sic_target_masked = np.ma.masked_array(sic_target, mask=lsmask)
    sic_pred_masked = np.ma.masked_array(sic_pred, mask=lsmask)

    iiee = IIEE(sic_pred_masked, sic_target_masked)

    A_plus = iiee[0].sum()
    A_minus = iiee[1].sum()

    A_iiee = A_plus + A_minus
    alpha_iiee = A_plus - A_minus

    D_iiee = (2 * A_iiee) / (length_target + length_pred)
    delta = (2 * alpha_iiee) / (length_target + length_pred)

    print(f"{D_iiee=}")
    print(f"{delta=}")





if __name__ == "__main__":
    main()