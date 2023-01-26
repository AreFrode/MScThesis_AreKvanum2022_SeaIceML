import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid")
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels")

import numpy as np

from calendar import monthrange
from netCDF4 import Dataset

from interpolate import nearest_neighbor_interp
from common_functions import onehot_encode_sic, get_target_domain

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Define paths
    path_barents = "/lustre/storeB/project/fou/hi/oper/barents_eps/archive/eps/"
    proj4_arome = "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06"

    # Use processed largest grid for regrid domain
    common_grid = sys.argv[2]

    path_output, transform_function, target_x, target_y, target_lat, target_lon = get_target_domain(common_grid, proj4_arome, 'barents')

    nx = len(target_x)
    ny = len(target_y)

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

        # Define baltic + norway mask
        baltic_mask = np.zeros((len(barents_y), len(barents_x)))
        baltic_mask[:520, 430:] = 1

        barents_sic = np.where(baltic_mask == 1, 0, barents_sic)
        
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

        mean_1 = np.mean(interp_array[1:7], axis=0, keepdims=True)
        mean_2 = np.mean(interp_array[7:13], axis=0, keepdims=True)
        mean_3 = np.mean(interp_array[13:19], axis=0, keepdims=True)

        interp_array = np.concatenate((interp_array, mean_1, mean_2, mean_3))

        interpolated = nearest_neighbor_interp(xxc_target, yyc_target, target_x, target_y, interp_array)
        
        sic_target_interp = interpolated[1:19].reshape((len(lead_times) - 1, n_members, ny, nx))

        output_filename = f"barents_mean_b{yyyymmdd}.nc"

        with Dataset(f"{path_output_task}{output_filename}", 'w', format = "NETCDF4") as nc_out:
            nc_out.createDimension('x', len(target_x))
            nc_out.createDimension('y', len(target_y))
            nc_out.createDimension('t', 3)
            nc_out.createDimension('member', 6)

            yc = nc_out.createVariable('y', 'd', ('y'))
            yc.units = 'km'
            yc.standard_name = 'y'
            yc[:] = target_y
            
            xc = nc_out.createVariable('x', 'd', ('x'))
            xc.units = 'km'
            xc.standard_name = 'x'
            xc[:] = target_x

            latc = nc_out.createVariable('lat', 'd', ('y', 'x'))
            latc.units = 'degrees North'
            latc.standard_name = 'Latitude'
            latc[:] = target_lat

            lonc = nc_out.createVariable('lon', 'd', ('y', 'x'))
            lonc.units = 'degrees East'
            lonc.standard_name = 'Lonitude'
            lonc[:] = target_lon

            sic_out = nc_out.createVariable('sic', 'd', ('t', 'member', 'y', 'x'))
            sic_out.units = "1"
            sic_out.standard_name = "Sea Ice Concentration"
            sic_out[:] = onehot_encode_sic(sic_target_interp)

            mean_sic_out = nc_out.createVariable('mean_sic', 'd', ('t', 'y', 'x'))
            mean_sic_out.units = "1"
            mean_sic_out.standard_name = "Ensemble Mean Sea Ice Concentration"
            mean_sic_out[:] = onehot_encode_sic(interpolated[19:])

            lsmask_out = nc_out.createVariable('lsmask', 'd', ('y', 'x'))
            lsmask_out.units = "1"
            lsmask_out.standard_name = "Land Sea Mask"
            lsmask_out[:] = interpolated[0]


if __name__ == "__main__":
    main()