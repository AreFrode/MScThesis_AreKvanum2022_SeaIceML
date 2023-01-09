import numpy as np

def onehot_encode_sic(sic):
    fast_ice = np.where(np.equal(sic, 1.), 6, 0)
    vcd_ice = np.where(np.logical_and(np.greater_equal(sic, .9), np.less(sic,1.)), 5, 0)
    cd_ice = np.where(np.logical_and(np.greater_equal(sic, .7), np.less(sic, .9)), 4, 0)
    od_ice = np.where(np.logical_and(np.greater_equal(sic, .4), np.less(sic, .7)), 3, 0)
    vod_ice = np.where(np.logical_and(np.greater_equal(sic, .1), np.less(sic, .4)), 2, 0)
    open_water = np.where(np.logical_and(np.greater(sic, 0.), np.less(sic, .1)), 1, 0)
    invalid_above = np.where(np.logical_or(np.isnan(sic), np.greater(sic, 1.)), -10, 0)
    invalid_below = np.where(np.less(sic, 0.), -10, 0)

    return fast_ice + vcd_ice + cd_ice + od_ice + vod_ice + open_water + invalid_above + invalid_below

def onehot_encode_sic_numerical(sic):
    fast_ice = np.where(np.equal(sic, 100.), 6, 0)
    vcd_ice = np.where(np.logical_and(np.greater_equal(sic, 90.), np.less(sic,100.)), 5, 0)
    cd_ice = np.where(np.logical_and(np.greater_equal(sic, 70.), np.less(sic, 90.)), 4, 0)
    od_ice = np.where(np.logical_and(np.greater_equal(sic, 40.), np.less(sic, 70.)), 3, 0)
    vod_ice = np.where(np.logical_and(np.greater_equal(sic, 10.), np.less(sic, 40.)), 2, 0)
    open_water = np.where(np.logical_and(np.greater(sic, 0.), np.less(sic, 10.)), 1, 0)

    return fast_ice + vcd_ice + cd_ice + od_ice + vod_ice + open_water