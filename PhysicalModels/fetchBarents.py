import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset
from pyproj import CRS, Transformer

from interpolate import nearest_neighbor_interp
from common_functions import onehot_encode_sic

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths
    path_barents = "/lustre/storeB/project/fou/hi/oper/barents_eps/archive/eps/"

    # Use processed nextsim for regrid domain
    path_target_nextsim = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim/2022/01/nextsim_mean_b20220101.nc"

    # Set boundaries from ml domain

    path_output = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/barents/"

    # Define projection transformer
    proj4_nextsim = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +x_0=0 +y_0=0 +R=6378273 +ellps=sphere +units=m +no_defs"
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    crs_NEXTSIM = CRS.from_proj4(proj4_nextsim)
    crs_AROME = CRS.from_proj4(proj4_arome)
    transform_function = Transformer.from_crs(crs_AROME, crs_NEXTSIM, always_xy = True)

    # Define target grid
    with Dataset(path_target_nextsim, 'r') as nc_ns:
        nextsim_x = nc_ns['x'][:]
        nextsim_y = nc_ns['y'][:]
        nextsim_lat = nc_ns['lat'][:]
        nextsim_lon = nc_ns['lon'][:]

    nx = len(nextsim_x)
    ny = len(nextsim_y)

    # Define months for parallel execution
    year = 2022
    months = []
    for month in range(1, 13):
        months.append(month)

    month_task = months[int(sys.argv[1]) - 1]
    print(f"{month_task=}")

    path_output_task = f"{path_output}{year}/{month_task:02d}/"
    print(f"{path_output_task=}")
    
    nb_days_task = monthrange(int(year), int(month_task))[1]
    print(f"{nb_days_task=}")

    if not os.path.exists(path_output_task):
        os.makedirs(path_output_task)

    # define start:stop intervals
    lead_times = [0, 24, 48, 66]

    for dd in range(1, nb_days_task + 1):
        yyyymmdd = f"{year}{month_task:02d}{dd:02d}"
        print(yyyymmdd)

        try:
            barents_path = glob.glob(f"{path_barents}barents_eps_{yyyymmdd}T00Z.nc")[0]

        except IndexError:
            print(f"Missing file b{yyyymmdd}")
            continue

        with Dataset(barents_path, 'r') as nc:
            barents_x = nc.variables['X'][:]
            barents_y = nc.variables['Y'][:]
            barents_sic = nc.variables['ice_concentration'][:,:,:,:]
            barents_lsmask = nc.variables['sea_mask'][0,:,:].astype(int)

        # Swap 1 and 0
        barents_lsmask = np.where((barents_lsmask==0)|(barents_lsmask==1), barents_lsmask^1, barents_lsmask)

        x_diff = barents_x[1] - barents_x[0]
        y_diff = barents_y[1] - barents_y[0]

        xc = np.pad(barents_x, (1,1), 'constant', constant_values = (barents_x[0] - x_diff, barents_x[-1] + x_diff))
        yc = np.pad(barents_y, (1,1), 'constant', constant_values = (barents_y[0] - y_diff, barents_y[-1] + y_diff))

        barents_sic_padded = np.pad(barents_sic, ((0,0), (0,0), (1,1), (1,1)), 'constant', constant_values = np.nan)
        barents_lsmask_padded = np.pad(barents_lsmask, ((1,1), (1,1)), 'constant', constant_values = 1)

        xxc, yyc = np.meshgrid(xc, yc)

        xxc_target, yyc_target = transform_function.transform(xxc, yyc)

        # Allocate target arrays
        n_members = barents_sic.shape[1]

        interp_list = [barents_lsmask_padded]

        for i in range(len(lead_times) - 1):
            for j in range(n_members):
                barents_sic = np.mean(barents_sic_padded[lead_times[i]:lead_times[i+1], j, ...], axis=0)
                interp_list.append(barents_sic)

        interp_array = np.array(interp_list)
        interpolated = nearest_neighbor_interp(xxc_target, yyc_target, nextsim_x, nextsim_y, interp_array)
        
        sic_target_interp = interpolated[1:].reshape((len(lead_times) - 1, n_members, ny, nx))

        output_filename = f"barents_mean_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(nextsim_x))
            nc_out.createDimension('y', len(nextsim_y))
            nc_out.createDimension('t', 3)
            nc_out.createDimension('member', 6)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = nextsim_y
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = nextsim_x

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees North'
            latc.standard_name = 'Latitude'
            latc[:] = nextsim_lat

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees East'
            lonc.standard_name = 'Lonitude'
            lonc[:] = nextsim_lon

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'member', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic(sic_target_interp)

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = interpolated[0]


if __name__ == "__main__":
    main()