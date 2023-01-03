# Helper-functions to regrid AA onto 1km grid
# Author: Cyril Palerme
# Date: 02.01.2023

# Regridding functions (nearest_neighbor_indexes and nearest_neighbor_interp)  

#     xx_input and yy_input must be 2D arrays
#     x_output and y_output must be vectors  
#     field must be either a 2D array with dimensions (y, x) or a 3D array with dimensions (time, y, x) 
#     invalid_values = fill value to replace by 0. Land is therefore considered as open ocean.

# This serves as an alternative to scipy griddata, since all data is on the same grid it speeds up calculations significantly

import numpy as np
from scipy.spatial import KDTree

def nearest_neighbor_indexes(x_input, y_input, x_output, y_output):
    x_input = np.expand_dims(x_input, axis = 1)
    y_input = np.expand_dims(y_input, axis = 1)
    x_output = np.expand_dims(x_output, axis = 1)
    y_output = np.expand_dims(y_output, axis = 1)
    #
    coord_input = np.concatenate((x_input, y_input), axis = 1)
    coord_output = np.concatenate((x_output, y_output), axis = 1)
    #
    tree = KDTree(coord_input)
    dist, idx = tree.query(coord_output)
    #
    return(idx)

def nearest_neighbor_interp(xx_input, yy_input, x_output, y_output, field, fill_value = None):
    xx_input_flat = np.ndarray.flatten(xx_input)
    yy_input_flat = np.ndarray.flatten(yy_input)
    #
    if fill_value is not None:
        if field.ndim == 2:
            idx_fill_value = np.ndarray.flatten(field) == fill_value
        elif field.ndim == 3:
            idx_fill_value = np.ndarray.flatten(field[0,:,:]) == fill_value
        #
        xx_input_flat = xx_input_flat[idx_fill_value == False]
        yy_input_flat = yy_input_flat[idx_fill_value == False]
    #
    xx_output, yy_output = np.meshgrid(x_output, y_output)
    xx_output_flat = np.ndarray.flatten(xx_output)
    yy_output_flat = np.ndarray.flatten(yy_output)
    #
    idx = nearest_neighbor_indexes(xx_input_flat, yy_input_flat, xx_output_flat, yy_output_flat)
    #
    if field.ndim == 2:
        field_flat = np.ndarray.flatten(field)
        if fill_value is not None:
            field_flat = field_flat[idx_fill_value == False]
        #
        field_interp = field_flat[idx]
        field_regrid = np.reshape(field_interp, (len(y_output), len(x_output)), order = "C")
    #    
    elif field.ndim == 3:
        time_dim = len(field[:,0,0])
        field_regrid = np.full((time_dim, len(y_output), len(x_output)), np.nan)
        #
        for t in range(0, time_dim):
            field_flat = np.ndarray.flatten(field[t,:,:])
            if fill_value is not None:
                field_flat = field_flat[idx_fill_value == False]
            #
            field_interp = field_flat[idx]
            field_regrid[t,:,:] = np.reshape(field_interp, (len(y_output), len(x_output)), order = "C")
    #
    return(field_regrid)